#!/usr/bin/env python
# -*- coding:utf-8-*-
# auther:kewei Liu time:2019/3/18 19:54  QQ:422209303  e-mail:Liukeweiaway@hotmail.com
# --------------------------------------------------------------------------
import pandas as pd  # 表格
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # k临近
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing  # 归一化
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn import naive_bayes  # 贝叶斯  多项式 伯努利     #决策树DT
from sklearn.svm import SVC  # 支持向量机
from sklearn.ensemble import RandomForestClassifier,  VotingClassifier  # voting
from datetime import datetime
import matplotlib.pyplot as plt

start_time = datetime.now()

RF_list_10 = []
knn_list_10 = []
naiveB_list_10 = []
svm_list_10 = []
LR_list_10 = []
eclf_list_10 = []

# mrmd2.0 80feature排序
# mrmr60个排序  # 37 0.8389
selection_feature_index = [53, 2, 10, 12, 48, 59, 13, 32, 56, 33, 45, 52, 43, 51, 54, 57, 60, 49, 42, 46, 58, 55, 34,
                           44, 29, 47, 23, 9, 31, 21, 22, 39, 28, 30, 50, 18, 26, 41, 37, 27, 35, 25, 38, 8, 40, 36, 24,
                           7, 14, 4, 20, 6, 16, 19, 1, 3, 17, 5, 15, 11]
length = len(selection_feature_index)  # 因为是从1开始就是两个特征开始，要减去1
for selection in range(1, len(selection_feature_index), 1):
    pddtrain = pd.read_csv("944_mm_samples_60_psxp_new_out", sep=' ', header=None)  # train data
    X_train = pddtrain.values[:, selection_feature_index[0:selection]]
    y_train = list(pddtrain.iloc[:, 0])
    # #------------------------------------------------------------------------------------------------------
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    # ##模型选择 分类器参数选择---------------------------------------------------------------------------------
    RF = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=9
                                , min_samples_split=12, min_samples_leaf=3, min_weight_fraction_leaf=0.06,
                                max_features='auto'
                                , max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                bootstrap=True
                                , oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False,
                                class_weight=None)
    knn = KNeighborsClassifier(n_neighbors=40, n_jobs=1)  # K临近算法
    naiveB = naive_bayes.BernoulliNB(alpha=1.6, binarize=1.41, fit_prior=True, class_prior=None)  # 0.575
    svm = SVC(C=10, kernel='rbf', gamma=0.001, probability=True)
    LR = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=4.0
                            , fit_intercept=True, intercept_scaling=2, class_weight=None
                            , random_state=None, solver='liblinear', max_iter=100, multi_class='ovr'
                            , verbose=0, warm_start=False, n_jobs=1)
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
# -----------------------------------------------------10cross最大值点的坐标-----------------------------
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

y_10c_acc_RF = np.array(RF_list_10)
y_10c_acc_knn = np.array(knn_list_10)
y_10c_acc_naiveB = np.array(naiveB_list_10)
y_10c_acc_svm = np.array(svm_list_10)
y_10c_acc_LR = np.array(LR_list_10)
y_10c_acc_eclf = np.array(eclf_list_10)
# ------------------------------------------------------------画图 2------------------------------
plt.figure(num=2, figsize=(8, 5))

plt.plot(x, y_10c_acc_RF, color='red', label=r'$Random\ Forest$ ')
plt.plot(x, y_10c_acc_knn, color='green', label=r'$Nearest\ Neighbors$ ')
plt.plot(x, y_10c_acc_naiveB, color='blue', label=r'$Naive\ Bayes$ ')
plt.plot(x, y_10c_acc_svm, color='yellow', label=r'$Support\ Vector Machines$ ')
plt.plot(x, y_10c_acc_LR, color='cyan', label=r'$Logistic\ Regression$ ')
plt.plot(x, y_10c_acc_eclf, color='black', label=r'$ensemble\ classifier$ ')
plt.scatter(feature_number_10, max(max_list_10), s=50, color='b')
plt.annotate((feature_number_10-1, round(max(max_list_10), 3)), xy=(feature_number_10, max(max_list_10)), xycoords='data',
             xytext=(-30, +30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
plt.xlim((0, length + 1))
plt.ylim((0.6, 1))
plt.legend(loc="lower right")
plt.xlabel('number of features')
plt.ylabel('10-cross-validation-accuracy')
plt.show()
print(datetime.now() - start_time)
