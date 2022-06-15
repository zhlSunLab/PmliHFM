import numpy as np
from keras.utils import np_utils
from keras.layers import Dense, Dropout,Flatten, TimeDistributed,concatenate,Input
from keras.layers import Convolution2D,MaxPooling2D,BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping
from metrics import Evaluation
from features import high_order_hot,hot
from model import *
import keras.backend as K
from keras.callbacks import LearningRateScheduler

def onehot_t(list,k):

 # set 'A': 0, 'T': 1, 'C': 2, 'G': 3
    onehotsequence_mirna = []
    onehotlabel = []
    onehotsequence_lncrna = []

    for LinePair in list:
        if k == 1:
            miRNAname, lncRNAname, miRNAsequence, lncRNAsequence,miRNAstructure, lncRNAstructure, label = LinePair.strip().split(',')
        else:
            miRNAname, lncRNAname, miRNAsequence, lncRNAsequence, label = LinePair.strip().split( ',')

        miRNAsequence = miRNAsequence.upper()
        miRNAsequence = miRNAsequence.replace('U', 'T')
        lncRNAsequence = lncRNAsequence.upper()
        lncRNAsequence = lncRNAsequence.replace('U', 'T')

        onehotsequence_m= hot(miRNAsequence,TotalSequenceLength_m)
        onehotsequence_l =high_order_hot(lncRNAsequence,TotalSequenceLength_l+2,nums)


        onehotsequence_mirna.append(onehotsequence_m)  
        onehotsequence_lncrna.append(onehotsequence_l)

        onehotlabel.append(label.strip('\n')) 


    X_miran= np.array(onehotsequence_mirna).reshape(-1,TotalSequenceLength_m, 4,1)
    X_miran = X_miran.astype('float32')     #保留小数
    X_lncran= np.array(onehotsequence_lncrna).reshape(-1,TotalSequenceLength_l, 64,1)
    X_lncran = X_lncran.astype('float32')     #保留小数
    Y = np.array(onehotlabel).astype('int').reshape(-1, 1)   #Y[20,1]
    Y = np_utils.to_categorical(Y, num_classes=2)       #Y[20,2]

    return X_miran, X_lncran,Y



def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model_ensemble.optimizer.lr)
        K.set_value(model_ensemble.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model_ensemble.optimizer.lr)

# ListData_train = 'Training-validation-dataset/premli/pemg-mli-3200.fasta'   #下载数据集
#ListData_train = 'Training-validation-dataset/premli/pemg-mli-3200.fasta'   #下载数据集
ListData_train = 'trainingdata/pemg_7680.fasta'   #下载数据集
#ListData_train = 'trainingdata/train_8000.fasta'   #下载数据集
ListTrain = open(ListData_train, 'r').readlines()


# ListData_test = 'Training-validation-dataset/premli/pemg-mli-800.fasta'   #下载数据集
# ListData_test = 'Training-validation-dataset/pmlipred/3000.fasta'   #下载数据集
#ListData_test = 'Training-validation-dataset/imbalance8-2/pemg_imb800.fasta'   #下载数据集
ListData_test = 'testdata/imbalance/im_test1056.fasta'   #下载数据集


ListTest = open(ListData_test, 'r').readlines()
TotalSequenceLength_m=0   
TotalSequenceLength_l=0   
num1 = 0
num2 =0

for LinePair in ListTrain:
    num1+=1
    miRNAname,lncRNAname,miRNAsequence,lncRNAsequence,miRNAstructure,lncRNAstructure,label = LinePair.strip().split(',')
    #miRNAname, lncRNAname, miRNAsequence, lncRNAsequence,label = LinePair.strip().split(',')
    if len(lncRNAsequence) > TotalSequenceLength_l:
        TotalSequenceLength_l = len(lncRNAsequence)
    if len(miRNAsequence) > TotalSequenceLength_m:
        TotalSequenceLength_m = len(miRNAsequence)

for LinePair in ListTest:
    num2+=1
   # miRNAname, lncRNAname, miRNAsequence, lncRNAsequence,miRNAstructure, lncRNAstructure,label = LinePair.strip().split(',')
    miRNAname, lncRNAname, miRNAsequence, lncRNAsequence,label = LinePair.strip().split(',')
    if len(lncRNAsequence) > TotalSequenceLength_l:
        TotalSequenceLength_l = len(lncRNAsequence)
    if len(miRNAsequence) > TotalSequenceLength_m:
        TotalSequenceLength_m = len(miRNAsequence)


print('##################### Load data  #####################\n')
X_miran_train, X_lncran_train,y_train = onehot_t(ListTrain,1)
X_miran_test, X_lncran_test,y_test = onehot_t(ListTest,2)

# np.random.seed(874)
# np.random.shuffle(X_miran_train)
# np.random.seed(874)
# np.random.shuffle(X_lncran_train)
# np.random.seed(874)
# np.random.shuffle(y_train)
# np.random.seed(35)
# np.random.shuffle(X_miran_test)
# np.random.seed(35)
# np.random.shuffle(X_lncran_test)
# np.random.seed(35)
# np.random.shuffle(y_test)

mirna_sam = SAM_BLOCK(TotalSequenceLength_m,4)     #X_train2, y_train2为要训练的数据和标签 合适
print('##################### mirna_sae completed #####################\n')
mirna_scm = SCM_BLOCK(TotalSequenceLength_m,4)
print('##################### mirna_cnn completed #####################\n')

lncrna_sam = SAM_BLOCK(TotalSequenceLength_l,64)
print('##################### lncrna_sae completed #####################\n')
lncrna_scm = SCM_BLOCK(TotalSequenceLength_l,64)
print('##################### lncrna_cnn completed #####################\n')

ensemble_in = concatenate([mirna_scm.output, lncrna_scm.output,
                           mirna_sam.output, lncrna_sam.output])

ensemble_in = Dropout(0.25)(ensemble_in)
ensemble = Dense(16, kernel_initializer='random_uniform', activation='relu')(ensemble_in)
ensemble = BatchNormalization()(ensemble)
ensemble = Dense(8, kernel_initializer='random_uniform', activation='relu')(ensemble)
ensemble = BatchNormalization()(ensemble)
ensemble_out = Dense(2, activation='softmax')(ensemble)
print('#################### Ensemble model completed####################\n ')
X_train20 = X_miran_train.reshape(num1,TotalSequenceLength_m*4 )
X_train30 = X_lncran_train.reshape(num1,TotalSequenceLength_l*64)
X_test20 = X_miran_test.reshape(num2,TotalSequenceLength_m*4)
X_test30 = X_lncran_test.reshape(num2,TotalSequenceLength_l*64)
model_ensemble = Model(inputs=[mirna_scm.input] + [lncrna_scm.input] + [mirna_sam.input] + [lncrna_sam.input], outputs=ensemble_out)
print('#################### model_ensemble completed####################\n ')
X_ensemble_train = [X_miran_train]+ [X_lncran_train] + [X_train20]+ [X_train30]  
X_ensemble_test =[X_miran_test] + [X_lncran_test]+ [X_test20]+ [X_test30] 


reduce_lr = LearningRateScheduler(scheduler)
adam = optimizers.Adam(lr=1e-3)
model_ensemble.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
es = EarlyStopping(monitor='val_lacc', mode='max', verbose=1, patience=10)  #

model_ensemble.fit(x=X_ensemble_train, y=[y_train],validation_data=(X_ensemble_test,y_test),epochs=30,batch_size=32,verbose=1,callbacks=[es,reduce_lr])

y_test_predict = model_ensemble.predict(X_ensemble_test)

model_ensemble.summary()
TP, FP, TN, FN, SEN, SPE, ACC, F1, AUC = Evaluation(y_test, y_test_predict)
print('##################### Evalucation completed #####################\n')

print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
print('TPR:', SEN, 'TNR:', SPE, 'ACC:', ACC, 'F1:', F1, 'AUC:', AUC)
