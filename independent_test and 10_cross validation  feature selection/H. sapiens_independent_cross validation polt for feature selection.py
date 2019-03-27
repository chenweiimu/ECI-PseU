#!/usr/bin/env python
# -*- coding:utf-8-*-
# auther:kewei Liu time:2019/3/18 19:39  QQ:422209303  e-mail:Liukeweiaway@hotmail.com
# --------------------------------------------------------------------------
import pandas as pd  # 表格
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # k临近
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing  # 归一化
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn import naive_bayes  # 贝叶斯  多项式 伯努利     #决策树DT
from sklearn.svm import SVC  # 支持向量机
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier  # voting
from datetime import datetime
from sklearn.metrics import roc_curve, auc  # AUC曲线
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef  # MCC相关系数
from scipy import interp

start_time = datetime.now()
rf_list = []
knn_list = []
naiveB_list = []
svm_list = []
LR_list = []
eclf_list = []
RF_list_10 = []
knn_list_10 = []
naiveB_list_10 = []
svm_list_10 = []
LR_list_10 = []
eclf_list_10 = []

# ##文件输入
# mrmd2.0 80feature排序

# mrmr60个排序   # 53 0.695
selection_feature_index = [53, 20, 31, 46, 57, 1, 9, 44, 12, 60, 54, 49, 59, 47, 33, 55, 42, 45, 58, 32,
                           56, 48, 43, 52, 27, 51, 29, 40, 30, 34, 24, 38, 22, 50, 39, 6, 10, 13, 41, 25,
                           35, 23, 18, 26, 36, 14, 19, 28, 37, 2, 21, 7, 5, 8, 17, 4, 15, 3, 16, 11]
length = len(selection_feature_index)  # 因为是从1开始就是两个特征开始，要减去1
for selection in range(1, len(selection_feature_index), 1):
    pddtrain = pd.read_csv("990samples_60_psxp_new_out", sep=' ', header=None)  # train data
    X_train = pddtrain.values[:, selection_feature_index[0:selection]]
    y_train = list(pddtrain.iloc[:, 0])
    pddtest = pd.read_csv("200samples_60_psxp_new_out", sep=' ', header=None)  # predict data
    X_test = pddtest.values[:, selection_feature_index[0:selection]]
    y_test = list(pddtest.iloc[:, 0])
    # #------------------------------------------------------------------------------------------------------
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # ##模型选择 分类器参数选择---------------------------------------------------------------------------------
    RF = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=8,
                                min_samples_split=12, min_samples_leaf=3, min_weight_fraction_leaf=0.06,
                                max_features='auto',
                                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False,
                                class_weight=None)
    knn = KNeighborsClassifier(n_neighbors=40, n_jobs=1)  # K临近算法
    naiveB = naive_bayes.BernoulliNB(alpha=1.6, binarize=1.41, fit_prior=True, class_prior=None)  # 0.575
    svm = SVC(C=1, kernel='rbf', gamma=0.001, probability=True)
    LR = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=4.0,
                            fit_intercept=True, intercept_scaling=2, class_weight=None,
                            random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                            verbose=0, warm_start=False, n_jobs=1)
    eclf = VotingClassifier(estimators=[('RF', RF), ('knn', knn), ('svm', svm), ('naiveB', naiveB), ('LR', LR)],
                            voting='soft', weights=[1, 1, 1, 1, 1])
    # # #训练模型---------------------------------------------------------------------------------------------
    RF.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    naiveB.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    eclf.fit(X_train, y_train)
    LR.fit(X_train, y_train)
    # -------------------------------------交叉验证
    RF_scores = cross_val_score(RF, X_train, y_train, cv=10, scoring='accuracy')
    knn_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    naiveB_scores = cross_val_score(naiveB, X_train, y_train, cv=10, scoring='accuracy')
    svm_scores = cross_val_score(svm, X_train, y_train, cv=10, scoring='accuracy')
    LR_list_scores = cross_val_score(LR, X_train, y_train, cv=10, scoring='accuracy')
    eclf_scores = cross_val_score(eclf, X_train, y_train, cv=10, scoring='accuracy')
    # # #测试集准确率-----------------------------------------------------------------------------------------
    print(selection)
    print('RF:', RF.score(X_test, y_test))
    print('knn:', knn.score(X_test, y_test))
    print('naiveB:', naiveB.score(X_test, y_test))
    print('SVM:', svm.score(X_test, y_test))
    print('LR:', LR.score(X_test, y_test))
    print('eclf:', eclf.score(X_test, y_test))

    RF_scores_t = RF.score(X_test, y_test)
    knn_scores_t = knn.score(X_test, y_test)
    naiveB_scores_t = naiveB.score(X_test, y_test)
    svm_scores_t = svm.score(X_test, y_test)
    LR_scores_t = LR.score(X_test, y_test)
    eclf_scores_t = eclf.score(X_test, y_test)

    rf_list.append(RF_scores_t)
    knn_list.append(knn_scores_t)
    naiveB_list.append(naiveB_scores_t)
    svm_list.append(svm_scores_t)
    LR_list.append(LR_scores_t)
    eclf_list.append(eclf_scores_t)

    RF_list_10.append(RF_scores.mean())
    knn_list_10.append(knn_scores.mean())
    naiveB_list_10.append(naiveB_scores.mean())
    svm_list_10.append(svm_scores.mean())
    LR_list_10.append(LR_list_scores.mean())
    eclf_list_10.append(eclf_scores.mean())
    # # #测试集准确率-----------------------------------------------------------------------------------------
print('rf_list_10:', max(RF_list_10), RF_list_10.index(max(RF_list_10)))
print('knn_list_10:', max(knn_list_10), knn_list_10.index(max(knn_list_10)))
print('naiveB_list_10:', max(naiveB_list_10), naiveB_list_10.index(max(naiveB_list_10)))
print('svm_list_10:', max(svm_list_10), svm_list_10.index(max(svm_list_10)))
print('LR_list_10:', max(LR_list_10), LR_list_10.index(max(LR_list_10)))
print('eclf_list_10:', max(eclf_list_10), eclf_list_10.index(max(eclf_list_10)))
# # #10_cross准确率-----------------------------------------------
print('rf_list:', max(rf_list), rf_list.index(max(rf_list)))
print('knn_list:', max(knn_list), knn_list.index(max(knn_list)))
print('naiveB_list:', max(naiveB_list), naiveB_list.index(max(naiveB_list)))
print('svm_list:', max(svm_list), svm_list.index(max(svm_list)))
print('LR_list:', max(LR_list), LR_list.index(max(LR_list)))
print('eclf_list:', max(eclf_list), eclf_list.index(max(eclf_list)))
# -----------------------------------------------------测试集最大值点的坐标-----------------------------
max_list = [max(rf_list), max(knn_list), max(naiveB_list), max(svm_list), max(LR_list), max(eclf_list)]
max_list.index(max(max_list))
feature_number = length
if max_list.index(max(max_list)) == 0:
    feature_number = rf_list.index(max(max_list))+2
elif max_list.index(max(max_list)) == 1:
    feature_number = knn_list.index(max(max_list))+2
elif max_list.index(max(max_list)) == 2:
    feature_number = naiveB_list.index(max(max_list))+2
elif max_list.index(max(max_list)) == 3:
    feature_number = svm_list.index(max(max_list))+2
elif max_list.index(max(max_list)) == 4:
    feature_number = LR_list.index(max(max_list))+2
elif max_list.index(max(max_list)) == 5:
    feature_number = eclf_list.index(max(max_list))+2
# -----------------------------------------------------测试集最大值点的坐标-----------------------------
max_list_10 = [max(RF_list_10), max(knn_list_10), max(naiveB_list_10), max(svm_list_10), max(LR_list_10),
               max(eclf_list_10)]
max_list_10.index(max(max_list_10))
feature_number_10 = length  # 这里的feature_number是index number。+2才是feature_number
if max_list_10.index(max(max_list_10)) == 0:
    feature_number_10 = RF_list_10.index(max(max_list_10))+2
elif max_list_10.index(max(max_list_10)) == 1:
    feature_number_10 = knn_list_10.index(max(max_list_10))+2
elif max_list_10.index(max(max_list_10)) == 2:
    feature_number_10 = naiveB_list_10.index(max(max_list_10))+2
elif max_list_10.index(max(max_list_10)) == 3:
    feature_number_10 = svm_list_10.index(max(max_list_10))+2
elif max_list_10.index(max(max_list_10)) == 4:
    feature_number_10 = LR_list_10.index(max(max_list_10))+2
elif max_list_10.index(max(max_list_10)) == 5:
    feature_number_10 = eclf_list_10.index(max(max_list_10))+2
# -----------------------------------------------------x y1 y2-----------------------------
x = np.array(range(len(selection_feature_index)-1))+2
y_idp_acc_RF = np.array(rf_list)
y_idp_acc_knn = np.array(knn_list)
y_idp_acc_naiveB = np.array(naiveB_list)
y_idp_acc_svm = np.array(svm_list)
y_idp_acc_LR = np.array(LR_list)
y_idp_acc_eclf = np.array(eclf_list)

y_10c_acc_RF = np.array(RF_list_10)
y_10c_acc_knn = np.array(knn_list_10)
y_10c_acc_naiveB = np.array(naiveB_list_10)
y_10c_acc_svm = np.array(svm_list_10)
y_10c_acc_LR = np.array(LR_list_10)
y_10c_acc_eclf = np.array(eclf_list_10)
# ------------------------------------------------------------画图 1------------------------------
plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y_idp_acc_RF, color='red', label=r'$Random\ Forest$ ')
plt.plot(x, y_idp_acc_knn, color='green', label=r'$Nearest\ Neighbors$ ')
plt.plot(x, y_idp_acc_naiveB, color='blue', label=r'$Naive\ Bayes$ ')
plt.plot(x, y_idp_acc_svm, color='yellow', label=r'$Support\ Vector Machines$ ')
plt.plot(x, y_idp_acc_LR, color='cyan', label=r'$Logistic\ Regression$ ')
plt.plot(x, y_idp_acc_eclf, color='black', label=r'$ensemble\ classifier$ ')
plt.xlim((0, length + 1))
plt.ylim((0.4, 0.8))
plt.scatter(feature_number, max(max_list), s=50, color='b')
plt.annotate((feature_number-1, round(max(max_list), 3)), xy=(feature_number, max(max_list)), xycoords='data', xytext=(-30, +30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
plt.legend(loc="lower right")
plt.xlabel('number of features')
plt.ylabel('independent-data-test-accuracy')
plt.show()
# ------------------------------------------------------------画图 2------------------------------
plt.figure(num=2, figsize=(8, 5))

plt.plot(x, y_10c_acc_RF, color='red', label=r'$Random\ Forest$ ')
plt.plot(x, y_10c_acc_knn, color='green', label=r'$Nearest\ Neighbors$ ')
plt.plot(x, y_10c_acc_naiveB, color='blue', label=r'$Naive\ Bayes$ ')
plt.plot(x, y_10c_acc_svm, color='yellow', label=r'$Support\ Vector Machines$ ')
plt.plot(x, y_10c_acc_LR, color='cyan', label=r'$Logistic\ Regression$ ')
plt.plot(x, y_10c_acc_eclf, color='black', label=r'$ensemble\ classifier$ ')
plt.scatter(feature_number_10, max(max_list_10), s=50, color='b')
plt.annotate((feature_number_10-1, round(max(max_list_10), 3)), xy=(feature_number_10-1, max(max_list_10)), xycoords='data',
             xytext=(-30, +30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
plt.xlim((0, length + 1))
plt.ylim((0.6, 1))
plt.legend(loc="lower right")
plt.xlabel('number of features')
plt.ylabel('10-cross-validation-accuracy')
plt.show()
# -----------------------------画轴----------------------------------------------

print(datetime.now() - start_time)
