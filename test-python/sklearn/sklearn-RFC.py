# 加载数据集
from sklearn import datasets
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


Cvalues = [1e-2, 1e-1, 1e0, 1e1, 1e2]
n_estimators_values = [10, 100, 1000]


# generate the dataset
dataset = datasets.make_classification(
    n_samples=2000, n_features=20, 
    n_informative=8, n_redundant=2, n_repeated=0, 
    n_classes=2,
    random_state=6
    )
all_X = dataset[0]
all_y = dataset[1]



def GaussianNB_model(X_train, y_train, X_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def rbf_svm_model(X_train, y_train, X_test, C):
    clf = SVC(C=C, kernel='rbf', class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def RandomForestClassifier_model(X_train, y_train, X_test, n_estimators):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)



scores_acc = []
scores_f1 = []
scores_auc = []

clf_model = [GaussianNB_model, GaussianNB_model, RandomForestClassifier_model]


for i in range(3):
    """ 
    0 : GaussianNB
    1 : SVC
    2 : RandomForest
    """
    cur_acc = []
    cur_f1 = []
    cur_auc = []
    kf = cross_validation.KFold(len(all_X), n_folds=10, shuffle=True)
    for train_index, test_index in kf:
        # get train set and test set
        X_train, y_train = all_X[train_index], all_y[train_index]
        X_test, y_test = all_X[test_index], all_y[test_index]
        # get y_predict in best parameter
        if i == 0:
            # 如果是GaussionNB，不需要选择参数，可以直接计算结果
            y_predict = GaussianNB_model(X_train, y_train, X_test)
        else:
            # train the model and get the best parameter
            innerscore = []
            # 对于另外两种模型，需要指定参数列表和模型函数
            if i == 1 :
                parameters = Cvalues
                local_model = rbf_svm_model
            elif i == 2:
                parameters = n_estimators_values
                local_model = RandomForestClassifier_model
            for parameter in parameters:
                # parameter seletor
                ikf = cross_validation.KFold(len(X_train), n_folds=5, shuffle=True, random_state=33)
                innerf1 = []
                for t_index, v_index in ikf:
                    X_t, X_v = X_train[t_index], X_train[v_index]
                    y_t, y_v = y_train[t_index], y_train[v_index]
                    i_y_predict = local_model(X_t, y_t, X_v, parameter)
                    innerf1.append(metrics.f1_score(y_v, i_y_predict))
                innerscore.append(np.mean(innerf1))
            # 经过一个循环后，每一个参数的分数都存放在了innerscore列表中，此后取分数最高的参数进行预测
            bestParameter = parameters[np.argmax(innerscore)]
            print(bestParameter)
            # use the trained model to predict the output
            y_predict = local_model(X_train, y_train, X_test, bestParameter)
        # calculate the predict result score
        print(y_test)
        print(y_predict)
        cur_acc.append(metrics.accuracy_score(y_test, y_predict))
        cur_f1.append(metrics.f1_score(y_test, y_predict, average='macro'))
        cur_auc.append(metrics.roc_auc_score(y_test, y_predict))
    # 计算每一种方法的最终得分
    scores_acc.append(np.mean(cur_acc))
    scores_f1.append(np.mean(cur_f1))
    scores_auc.append(np.mean(cur_auc))

print(scores_acc, scores_f1, scores_auc)


name_list = ['Accuracy','F1-score','AUC ROC']  
x =list(range(3))  
total_width, n = 0.9, 3

width = total_width / n    
plt.bar(x, scores_acc, width=width, label='GaussionNB',fc = 'y')  
for i in range(len(x)):  
    x[i] = x[i] + width  
plt.bar(x, scores_f1, width=width, label='SVC', tick_label = name_list,fc = 'b')  
for i in range(len(x)):  
    x[i] = x[i] + width  
plt.bar(x, scores_auc, width=width, label='RandomForest',fc = 'r')  
plt.legend()  
plt.show()  
