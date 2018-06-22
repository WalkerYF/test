# 加载数据集
from sklearn import datasets
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def rbf_svm(X_train, y_train, X_test, C):
    clf = SVC(C=C, kernel='rbf', class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)



# generate the dataset
dataset = datasets.make_classification(
    n_samples=1000, n_features=10, 
    n_informative=2, n_redundant=2, 
    n_repeated=0, n_classes=2,
    random_state=3
    )
all_X = dataset[0]
all_y = dataset[1]


scores_index = range(1,11)
scores_acc = []
scores_f1 = []
scores_auc = []

Cvalues = [1e-2, 1e-1, 1e0, 1e1, 1e2]
n_estimators_values = [10, 100, 1000]

# get the index
kf = cross_validation.KFold(len(all_X), n_folds=10, shuffle=True)
for train_index, test_index in kf:
    print(train_index, test_index)
    # get train set and test set
    X_train, y_train = all_X[train_index], all_y[train_index]
    X_test, y_test = all_X[test_index], all_y[test_index]
    # train the model and get the best C
    innerscore = []
    for C in Cvalues:
        # parameter seletor
        ikf = cross_validation.KFold(len(X_train), n_folds=5, shuffle=True, random_state=33)
        innerf1 = []
        for t_index, v_index in ikf:
            X_t, X_v = X_train[t_index], X_train[v_index]
            y_t, y_v = y_train[t_index], y_train[v_index]

            i_y_predict = rbf_svm(X_t, y_t, X_v, C)
            innerf1.append(metrics.f1_score(y_v, i_y_predict))
        innerscore.append(np.mean(innerf1))
    bestC = Cvalues[np.argmax(innerscore)]
    # use the trained model to predict the output
    y_predict = rbf_svm(X_train, y_train, X_test, bestC)
    # calculate the predict result score
    scores_acc.append(metrics.accuracy_score(y_test, y_predict))
    scores_f1.append(metrics.f1_score(y_test, y_predict))
    scores_auc.append(metrics.roc_auc_score(y_test, y_predict))

# print(np.mean(scores))
plt.plot(scores_index, scores_acc)
plt.plot(scores_index, scores_f1)
plt.plot(scores_index, scores_auc)
plt.show()


