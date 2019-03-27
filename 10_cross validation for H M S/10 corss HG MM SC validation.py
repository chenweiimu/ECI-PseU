#!/usr/bin/env python
# -*- coding:utf-8-*-
# auther:kewei Liu time:2019/3/2 15:30  QQ:422209303  e-mail:Liukeweiaway@hotmail.com
# --------------------------------------------------------------------------
import pandas as pd  # 表格
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # k临近
from sklearn import preprocessing  # 归一化
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn import naive_bayes  # 贝叶斯  多项式 伯努利     #决策树DT
from sklearn.svm import SVC  # 支持向量机
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # voting
from datetime import datetime
from sklearn.metrics import roc_curve, auc  # AUC曲线
from sklearn.model_selection import StratifiedKFold, KFold
from scipy import interp
import matplotlib.pyplot as plt

start_time = datetime.now()
# #--------------------------HG----数据集------------------------------------------------------
selection_feature_index_hg = [53, 20, 31, 46, 57, 1, 9, 44, 12, 60, 54, 49, 59, 47, 33, 55, 42, 45, 58, 32, 56, 48, 43, 52,
                           27, 51, 29, 40, 30, 34, 24, 38, 22, 50, 39, 6, 10, 13, 41, 25, 35, 23, 18, 26, 36, 14, 19,
                           28, 37, 2, 21, 7, 5, 17, 4]
pddtrainHG = pd.read_csv("990samples_60_psxp_new_out", sep=' ', header=None)  # train data
X_trainHG = pddtrainHG.values[:, selection_feature_index_hg]
y_trainHG = list(pddtrainHG.iloc[:, 0])
# #--------------------------MM
selection_feature_index_mm = [53, 2, 10, 12, 48, 59, 13, 32, 56, 33, 45, 52, 43, 51, 54, 57, 60, 49, 42, 46,
                           58, 55, 34, 44, 29, 47, 23, 9, 31, 21, 22, 39, 28, 30, 50, 18, 26, 41]
pddtrainMM = pd.read_csv("944_mm_samples_60_psxp_new_out", sep=' ', header=None)  # train data
X_trainMM = pddtrainMM.values[:, selection_feature_index_mm]
y_trainMM = list(pddtrainMM.iloc[:, 0])
# #-------------------------SC
selection_feature_index_SC = [74, 18, 66, 82, 63, 73, 78, 70, 85, 79, 14, 88, 72, 68, 75, 64, 69, 81, 89, 45, 83, 62, 77,
                           67, 71, 44, 86, 90, 76, 80, 87, 84, 65, 48, 1, 43, 34, 40, 49, 32, 15, 30]
pddtrainSC = pd.read_csv("627_sc_samples_90_psxp_new_out", sep=' ', header=None)  # train data
X_trainSC = pddtrainSC.values[:, selection_feature_index_SC]
y_trainSC = list(pddtrainSC.iloc[:, 0])
# #-------------------------数据预处理------------------------------------------------------
X_trainHG = preprocessing.scale(X_trainHG)
X_trainMM = preprocessing.scale(X_trainMM)
X_trainSC = preprocessing.scale(X_trainSC)
# ##模型选择分类器参数选择------HG-----------------------------------------------------------------------------
RFHG = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=7
                              , min_samples_split=12, min_samples_leaf=3, min_weight_fraction_leaf=0.06,
                              max_features='auto'
                              , max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                              bootstrap=True
                              , oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False,
                              class_weight=None)
knnHG = KNeighborsClassifier(n_neighbors=40, n_jobs=1)  # K临近算法
naiveBHG = naive_bayes.GaussianNB(var_smoothing=0.01, priors=None)  # 0.575
svmHG = SVC(C=1, kernel='rbf', gamma=0.001, probability=True)
LRHG = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.001
                          , fit_intercept=True, intercept_scaling=2, class_weight=None
                          , random_state=None, solver='liblinear', max_iter=100, multi_class='ovr'
                          , verbose=0, warm_start=False, n_jobs=1)
eclfHG = VotingClassifier(estimators=[('RF', RFHG), ('knn', knnHG), ('svm', svmHG), ('naiveB', naiveBHG), ('LR', LRHG)],
                          voting='soft', weights=[0., 3, 0.0, 0.000, 0.0000])
# ##模型选择分类器参数选择------MM-
RFMM = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=9
                              , min_samples_split=12, min_samples_leaf=3, min_weight_fraction_leaf=0.06,
                              max_features='auto'
                              , max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                              bootstrap=True
                              , oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False,
                              class_weight=None)
knnMM = KNeighborsClassifier(n_neighbors=40, n_jobs=1)  # K临近算法
naiveBMM = naive_bayes.GaussianNB(var_smoothing=0.01, priors=None)  # 0.575
svmMM = SVC(C=10, kernel='rbf', gamma=0.001, probability=True)
LRMM = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=4.0
                          , fit_intercept=True, intercept_scaling=2, class_weight=None
                          , random_state=None, solver='liblinear', max_iter=100, multi_class='ovr'
                          , verbose=0, warm_start=False, n_jobs=1)
eclfMM = VotingClassifier(estimators=[('RF', RFMM), ('knn', knnMM), ('svm', svmMM), ('naiveB', naiveBMM), ('LR', LRMM)],
                          voting='soft', weights=[0.001, 0.001, 3, 0.001, 0.01])
# ##模型选择 分类器参数选择------SC-
RFSC = RandomForestClassifier(n_estimators=320, criterion='gini', max_depth=8
                              , min_samples_split=12, min_samples_leaf=3, min_weight_fraction_leaf=0.06,
                              max_features='auto'
                              , max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True
                              , oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False,
                              class_weight=None)
knnSC = KNeighborsClassifier(n_neighbors=40, n_jobs=1)  # K临近算法
naiveBSC = naive_bayes.GaussianNB(var_smoothing=0.01, priors=None)  # 0.575

svmSC = SVC(C=100, kernel='rbf', gamma=0.01, probability=True)
LRSC = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=4.0
                          , fit_intercept=True, intercept_scaling=2, class_weight=None
                          , random_state=None, solver='liblinear', max_iter=100, multi_class='ovr'
                          , verbose=0, warm_start=False, n_jobs=1)
eclfSC = VotingClassifier(estimators=[('RF', RFSC), ('knn', knnSC), ('naiveB', naiveBSC), ('svm', svmSC), ('LR', LRSC)],
                          voting='soft', weights=[5, 0.08, 0.0001, 0.001, 0.04])
# #-----------------------------------------cross validition-分层法---------------------------------------
sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
# #------------------------------------------------------------------------------------------------------
tprsHG = []
aucsHG = []
mean_fprHG = np.linspace(0, 1, 20)
tprsMM = []
aucsMM = []
mean_fprMM = np.linspace(0, 1, 20)
tprsSC = []
aucsSC = []
mean_fprSC = np.linspace(0, 1, 20)
# #---------------HG---------------fpr, tpr坐标确定  AUC确定--------------------------------------------------------
for train, test in sfolder.split(X_trainHG, y_trainHG):
    probas_HG = eclfHG.fit(np.array(X_trainHG)[train], np.array(y_trainHG)[train]).predict_proba(
        np.array(X_trainHG)[test])
    fpr, tpr, thresholds = roc_curve(np.array(y_trainHG)[test], probas_HG[:, 1])
    tprsHG.append(interp(mean_fprHG, fpr, tpr))
    tprsHG[-1][0] = 0.0
    roc_aucHG = auc(fpr, tpr)
    aucsHG.append(roc_aucHG)
# #---------------MM---
for train, test in sfolder.split(X_trainMM, y_trainMM):
    probas_MM = eclfMM.fit(np.array(X_trainMM)[train], np.array(y_trainMM)[train]).predict_proba(
        np.array(X_trainMM)[test])
    fpr, tpr, thresholds = roc_curve(np.array(y_trainMM)[test], probas_MM[:, 1])
    tprsMM.append(interp(mean_fprMM, fpr, tpr))
    tprsMM[-1][0] = 0.0
    roc_aucMM = auc(fpr, tpr)
    aucsMM.append(roc_aucMM)
# #---------------SC---
for train, test in sfolder.split(X_trainSC, y_trainSC):
    probas_SC = eclfSC.fit(np.array(X_trainSC)[train], np.array(y_trainSC)[train]).predict_proba(
        np.array(X_trainSC)[test])
    fpr, tpr, thresholds = roc_curve(np.array(y_trainSC)[test], probas_SC[:, 1])
    tprsSC.append(interp(mean_fprSC, fpr, tpr))
    tprsSC[-1][0] = 0.0
    roc_aucSC = auc(fpr, tpr)
    aucsSC.append(roc_aucSC)
# ----------------------------HG-----------------------------------------------
mean_tprHG = np.mean(tprsHG, axis=0)
mean_tprHG[-1] = 1.0
mean_aucHG = auc(mean_fprHG, mean_tprHG)
std_aucHG = np.std(aucsHG)
plt.plot(mean_fprHG, mean_tprHG, color='b',
         label=r'$H.\ sapiens$ (AUC = %0.2f)' % (mean_aucHG),
         lw=1, alpha=.8)
# ----------------------------MM
mean_tprMM = np.mean(tprsMM, axis=0)
mean_tprMM[-1] = 1.0
mean_aucMM = auc(mean_fprMM, mean_tprMM)
std_aucMM = np.std(aucsMM)
plt.plot(mean_fprMM, mean_tprMM, color='g',
         label=r'$M.\ musculus$ (AUC = %0.2f)' % (mean_aucMM),
         lw=1, alpha=.8)
# ----------------------------SC
mean_tprSC = np.mean(tprsSC, axis=0)
mean_tprSC[-1] = 1.0
mean_aucSC = auc(mean_fprSC, mean_tprSC)
std_aucSC = np.std(aucsSC)
plt.plot(mean_fprSC, mean_tprSC, color='r',
         label=r'$S.\ cerevisiae$ (AUC = %0.2f)' % (mean_aucSC),
         lw=1, alpha=.8)
# ------------------------------------------------------------------------------
plt.plot([0, 1], [0, 1], linestyle='-', lw=0.5, color='k', label='Chance', alpha=.8)  # 对角线
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
# plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.grid()
# ------------------------------------------------------------------------------
print(datetime.now() - start_time)
plt.show()
