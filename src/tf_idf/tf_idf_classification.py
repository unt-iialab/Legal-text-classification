import os
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocessing(dirpath, savepath):
    documents = []
    document_ids = []
    labels = []
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    # print(stop_words)
    pathDir = os.listdir(dirpath)
    # print(pathDir)
    for file in pathDir:
        document_ids.append(file)
        filename = file.split("_")
        label = filename[0]
        labels.append(label)
        # print(filename)
        filepath = os.path.join(dirpath,file)
        f = open(filepath)
        line = f.read()
        word_tokens = word_tokenize(line)
        # print(word_tokens)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        # print(filtered_sentence)
        pure_text = []
        for w in filtered_sentence:
            # print(w, " : ", ps.stem(w))
            pure_text.append(ps.stem(w))
        # print(pure_text)
        textstring = ' '.join([str(elem) for elem in pure_text])
        text_without_single = re.sub(r"\b[a-zA-Z]\b", "", textstring)
        documents.append(text_without_single)
    return documents, labels

# no feature selection
# tf-idf-RandomForest
# Accuracy: 0.6310013717421125
def tf_idf_RFclassification(document_list,labels):
    vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 3), min_df=0.01, max_df=0.8,
                                 stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(document_list).toarray()
    Y = labels
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    print(type(X))
    print(X)
    print(type(Y))
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    classifier = RandomForestClassifier(n_estimators=3000, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

# no feature selection
# four model: RandomForestClassifier,LinearSVC,MultinomialNB,LogisticRegression
def multimodels(document_list,labels):
    unique_labels = list(set(labels))
    vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 3), min_df=0.01, max_df=0.8,
                                 stop_words=stopwords.words('english'))
    features = vectorizer.fit_transform(document_list).toarray()
    X = vectorizer.fit_transform(document_list).toarray()
    Y = labels
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = MultinomialNB().fit(X_train, y_train)

    models = [
        RandomForestClassifier(n_estimators=5000, random_state=0),
        LinearSVC(max_iter=5000),
        MultinomialNB(),
        LogisticRegression(solver='lbfgs', multi_class='auto',random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=3)
    plt.show()
    performance = cv_df.groupby('model_name').accuracy.mean()
    print(performance)

    # prediction, we select RandomForestClassifier as the model
    classifier = RandomForestClassifier(n_estimators=3000, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, linewidths=2)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    print(conf_mat)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    dirpath = '/home/isiia/PycharmProjects/legal_classification/data_collection/all_courtlistener_clean_text'
    savepath = ''
    document_list, labels = preprocessing(dirpath, savepath)
    tf_idf_RFclassification(document_list,labels)
    # multimodels(document_list, labels)