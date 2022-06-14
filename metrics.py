import numpy as np
from sklearn.metrics import roc_auc_score

def Evaluation(y_test, resultslabel):  
    TP, FP, TN, FN = 0, 0, 0, 0
    AUC = roc_auc_score(y_test, resultslabel)   
    for row1 in range(resultslabel.shape[0]):       
        for column1 in range(resultslabel.shape[1]):   
            if resultslabel[row1][column1] < 0.5:    
                resultslabel[row1][column1] = 0
            else:
                resultslabel[row1][column1] = 1   
    for row2 in range(y_test.shape[0]):
        if y_test[row2][0] == 0 and y_test[row2][1] == 1 and y_test[row2][0] == resultslabel[row2][0] and y_test[row2][1] == resultslabel[row2][1]:
            TP = TP + 1     #TP是正确预测miRNA和lncRNA相互作用的个数
        if y_test[row2][0] == 1 and y_test[row2][1] == 0 and y_test[row2][0] != resultslabel[row2][0] and y_test[row2][1] != resultslabel[row2][1]:
            FP = FP + 1     #FP是错误预测miRNA和lncRNA之间不存在相互作用的个数
        if y_test[row2][0] == 1 and y_test[row2][1] == 0 and y_test[row2][0] == resultslabel[row2][0] and y_test[row2][1] == resultslabel[row2][1]:
            TN = TN + 1     #TN是正确预测miRNA和lncRNA之间不存在相互作用的个数
        if y_test[row2][0] == 0 and y_test[row2][1] == 1 and y_test[row2][0] != resultslabel[row2][0] and y_test[row2][1] != resultslabel[row2][1]:
            FN = FN + 1     #FN是错误预测miRNA和lncRNA相互作用的个数
    if TP + FN != 0:
        SEN = TP / (TP + FN)
    else:
        SEN = 999999
    if TN + FP != 0:
        SPE = TN / (TN + FP)
    else:
        SPE = 999999
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 999999
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 999999
    return TP, FP, TN, FN, SEN, SPE, ACC, F1, AUC


