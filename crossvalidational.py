# PmliPEMG for cross validation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import *
from features import *
from metrics import *

from keras.layers.embeddings import Embedding

def Kfold(X, Y, iteration, K):     
    totalpartX = len(X)
    partX = int(totalpartX / K)    
    totalpartY = len(Y)
    partY = int(totalpartY / K) 
    partXstart = iteration * partX  
    partXend = partXstart + partX  
    partYstart = iteration * partY 
    partYend = partYstart + partY   
    traindataP = np.array(X[0 : partXstart])
    traindataL = np.array(X[partXend : totalpartX]) 
    traindata = np.concatenate((traindataP, traindataL))   
    testdata = np.array(X[partXstart : partXend])  
    trainlabelP = np.array(Y[0 : partYstart])
    trainlabelL = np.array(Y[partYend : totalpartY])   
    trainlabel = np.concatenate((trainlabelP, trainlabelL))   
    testlabel = np.array(Y[partYstart : partYend])     
    return traindata, trainlabel, testdata, testlabel


ListData_test = 'Training-validation-dataset/pmlipemg/7680.fasta'  
ListTrain = open(ListData_test, 'r').readlines()
TotalSequenceLength_m=0   
TotalSequenceLength_l=0 
num = 0

for LinePair in ListTrain:     
    num+=1
    miRNAname, lncRNAname, miRNAsequence, lncRNAsequence,miRNAstructure, lncRNAstructure,label = LinePair.strip().split(',')
    if len(lncRNAsequence) > TotalSequenceLength_l:
        TotalSequenceLength_l = len(lncRNAsequence)
    if len(miRNAsequence) > TotalSequenceLength_m:
        TotalSequenceLength_m = len(miRNAsequence)

print('##################### Load data completed #####################\n')
X_miran, X_lncran,y = onehot(ListTrain)     
print('##################### onehot completed #####################\n')
TPsum, FPsum, TNsum, FNsum, SENsum, SPEsum, ACCsum, F1sum, AUCsum = [], [], [], [], [], [], [], [], []
for iteration in range(10):     #做十次10折交叉验证取平均值
    X_train2, y_train2, X_test2, y_test = Kfold(X_miran, y, iteration, 10)                                                                                #X_train2(8640,3959,4,1 )  y_train2(8640,2)
    mirna_sam = SAM_BLOCK(TotalSequenceLength_m,4)     
    print('##################### mirna_cnn completed #####################\n')
    mirna_scm = SCM_BLOCK(TotalSequenceLength_m,4)
    print('##################### mirna_sae completed #####################\n')
    X_train3, y_train3, X_test3, y_test = Kfold(X_lncran, y, iteration, 10)     
    lncrna_sam = SAM_BLOCK(TotalSequenceLength_l,64)
    print('##################### lncrna_cnn completed #####################\n')
    lncrna_scm = SCM_BLOCK(TotalSequenceLength_l,64)
    print('##################### lncrna_sae completed #####################\n')
    ensemble_in = concatenate([mirna_scm.output, lncrna_scm.output,
                               mirna_sam.output, lncrna_sam.output])
    ensemble_in = Dropout(0.25)(ensemble_in)
    ensemble = Dense(16, kernel_initializer='random_uniform', activation='relu')(ensemble_in)
    ensemble = BatchNormalization()(ensemble)
    ensemble = Dense(8, kernel_initializer='random_uniform', activation='relu')(ensemble)
    ensemble = BatchNormalization()(ensemble)
    ensemble_out = Dense(2, activation='softmax')(ensemble)
    print('#################### Ensemble model completed####################\n ')
    X_train20 = X_train2.reshape(6912,TotalSequenceLength_m*4)       
    X_train30 = X_train3.reshape(6912,TotalSequenceLength_l*64)       
    X_test20 = X_test2.reshape(768,TotalSequenceLength_m*4)          
    X_test30 = X_test3.reshape(768,TotalSequenceLength_l*64)        
    model_ensemble = Model(inputs=[mirna_scm.input] + [lncrna_scm.input] + [mirna_sam.input] + [lncrna_sam.input], outputs=ensemble_out)
    print('#################### model_ensemble completed####################\n ')
    X_ensemble_train = [X_train2]+ [X_train3] + [X_train20]+ [X_train30]   
    X_ensemble_test =[X_test2] + [X_test3]+ [X_test20]+ [X_test30]  

    import keras.backend as K
    from keras.callbacks import LearningRateScheduler

    #
    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(model_ensemble.optimizer.lr)
            K.set_value(model_ensemble.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model_ensemble.optimizer.lr)


    reduce_lr = LearningRateScheduler(scheduler)
    adam = optimizers.Adam(lr=1e-3)

    model_ensemble.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)  #

    model_ensemble.fit(x=X_ensemble_train, y=[y_train2],validation_data = (X_ensemble_test, y_test),epochs=30,batch_size=64,verbose=1,callbacks=[es,reduce_lr])

    # model_ensemble.summary()
    y_test_predict = model_ensemble.predict(X_ensemble_test)
    # model_ensemble.save(model_path + datetime.now().strftime("%Y%m%d-%H%M%S") + h + ".h5")
    print('##################### Ensemble model completed #####################\n')
    TP, FP, TN, FN, SEN, SPE, ACC, F1, AUC = Evaluation(y_test, y_test_predict)
    print('##################### Evalucation completed #####################\n')

    print('The ' + str(iteration + 1) + '-fold cross validation result')
    print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
    print('TPR:', SEN, 'TNR:', SPE, 'ACC:', ACC, 'F1:', F1, 'AUC:', AUC)

    TPsum.append(TP)
    FPsum.append(FP)
    TNsum.append(TN)
    FNsum.append(FN)
    SENsum.append(SEN)
    SPEsum.append(SPE)
    ACCsum.append(ACC)
    F1sum.append(F1)
    AUCsum.append(AUC)

# print the average results
print('The average results')
print('\ntest mean TPR: ', np.mean(SENsum))
print('\ntest mean TNR: ', np.mean(SPEsum))
print('\ntest mean ACC: ', np.mean(ACCsum))
print('\ntest mean F1-score: ', np.mean(F1sum))
print('\ntest mean AUC: ', np.mean(AUCsum))

