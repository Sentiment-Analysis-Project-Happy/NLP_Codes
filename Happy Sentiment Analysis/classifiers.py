# This file is used to fit different type od classifiers on the featureset generated through tf and tfidf.
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#from FeatureGeneration import create_feature_set_and_labels
from FeatureGenerationTFIDF import get_features


def begin_test(train_x, test_x, train_y, test_y):
    x = train_x + test_x
    y = train_y + test_y

    clf1 = LinearRegression()
    clf2 = LogisticRegression()
    clf3 = SGDClassifier()
    clf4 = SVC()
    clf5 = KNeighborsClassifier()
    clf6 = MLPClassifier()
    clf7 = DecisionTreeClassifier()
    clf8 = MultinomialNB()
    # clf1.fit(train_x, train_y)
    # y_pred = clf1.predict(test_x)
    # print("LinearRegressionAccuracy   ", accuracy_score(test_y, y_pred.round()))
    eclf = VotingClassifier(
        estimators=[('logr', clf2), ('sgd', clf3), ('svm', clf4), ('kn', clf5), ('nn', clf6), ('dt', clf7)],
        voting='hard')

    # for label, clf in zip(
    #         ['LogisticRegressionClassifier', 'SGDClassifierClassifier', 'SVCClassifier',
    #          'NearestNeighbourClassifier', 'NeuralNetworkClassifier', 'DecisionTreeClassifier',
    #          'MultinomialNB', 'EnsembleClassifier'],
    #         [clf2, clf3, clf4, clf5, clf6, clf7, clf8, eclf]):

    arr = [[16,54],[18,45],[23,54],[33,54],[37,45]]

    
    for k,j in arr:
            clf = DecisionTreeClassifier(splitter='best',max_depth=j,min_samples_split=k)
            clf.fit(train_x,train_y)
            y_pred = clf.predict(test_x)
            correct_count = 0
            for i in range(len(y_pred)):
                if(y_pred[i] == test_y[i]):
                    correct_count+=1
            # if(correct_count/len(y_pred)>=0.74):
            print(str(q)+"    "+str(k)+"    "+str(j)+"   "+str(correct_count/len(y_pred)))

    # for y1,y2 in test_y,y_pred:
    #     print(str(y1)+"     "+str(y2))
    
    # scores = cross_val_score(clf3, x, y, cv=10, scoring='accuracy')
    # f_measure = cross_val_score(clf3, x, y, cv=10, scoring='f1')
    #     # print(scores)
    # print('SGDClassifierClassifier', "Accuracy:  ", scores.mean(), "+/- ", scores.std())
    # print('SGDClassifierClassifier', "F-measure:  ", f_measure.mean())


# def test_by_tf():
#     train_x, train_y, test_x, test_y = create_feature_set_and_labels \
#         ('pos_hindi.txt', 'neg_hindi.txt')
#     begin_test(train_x, test_x, train_y, test_y)


def test_by_tfidf():
    train_x, train_y, test_x, test_y = get_features()
    begin_test(train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    # print("="*10)
    # print("Unigram+Tf Accuracies")
    # test_by_tf()
    print("=" * 10)
    print("Unigram+Tfidf Accuracies")
    test_by_tfidf()
