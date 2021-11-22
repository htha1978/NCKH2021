#Cac thu vien tien xu ly du lieu
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv
import re
import numpy as np
import time

# Cac thu vien, mo hinh multi-label classification
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from skmultilearn.problem_transform import ClassifierChain

from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix


# Cac thu vien de danh gia mo hinh
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#1.Cleaning data
def greeting(name):
    print("Hello, " + name)
    
def cleanData(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def stemming_sms_language(text):
    with open("data/sms_language_new.txt",encoding="utf-8") as f:
        for line in f:
          arr = line.rstrip('\n').split(',')
          text = re.sub(r"\b(" + arr[0] + r")\b", arr[1], text)
        f.close()
    #additions
    #text = re.sub(r"\b(usd|vnđ|vnd|)\b \b(0tr|0.tr |0.đ |0k|000d|00đ)\b", " đồng ", text) ## Xử lý tiền tệ
    #text = re.sub(r"\b(tcqc|tc qc|tu.choi)\b  \btu choi\b[ nhan ]*\b(tin|quang cao|qc|nhan tin|q.c)\b", "tu choi quang cao", text)
    return(text)

def data_preprocessing(data):
    #Chuyển về chữ thường
    data['Message'] = data['Message'].str.lower()
    data['Message'] = data['Message'].apply(stemming_sms_language)
    data['Message'] = data['Message'].apply(cleanData)
    data['Message'] = data['Message'].apply(cleanPunc)
    
    return(data)

def vectorit(data):
    train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)
    train_text = train['Message']
    test_text = test['Message']

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(train_text)
    vectorizer.fit(test_text)

    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['STT','Message'], axis=1)

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['STT','Message'], axis=1)
    
    return [vectorizer, x_train, y_train, x_test, y_test, train_text, test_text]

def printscoretable(y_test, predictions, walltimes):
    tbl = "<table border=1 cellspacing=0 cellpadding=0 class='dataframe'>"
    th = '<tr><th align=center>STT</th><th align=center>Tên tiêu chí</th><th align=center>Kết quả phân lớp</th></tr>'
    tr = ''
    # scores by samples
    tr+='<tr><td>1</td><td>Hamming Loss</td><td>'+str(round(hamming_loss(y_test,predictions),3))+'</td></tr>'
    tr+='<tr><td>2</td><td>One error</td><td>'+str(round(zero_one_loss(y_test,predictions),3))+'</td></tr>'
    y_pred = predictions.toarray()
    tr+='<tr><td>3</td><td>Coverage Error</td><td>'+str(round(coverage_error(y_test,y_pred),3))+'</td></tr>'
    tr+='<tr><td>4</td><td>label_ranking_loss</td><td>'+str(round(label_ranking_loss(y_test,y_pred),3))+'</td></tr>'
    tr+='<tr><td>5</td><td>Average Precision Score</td><td>'+str(round(average_precision_score(y_test,y_pred, average='micro'),3))+'</td></tr>'

    # scores by labels
    tr+='<tr><td>6</td><td>Accuracy Score</td><td>'+str(round(accuracy_score(y_test,predictions),3))+'</td></tr>'                                    
    tr+='<tr><td>7</td><td>Precision Score</td><td>'+str(round(precision_score(y_test,predictions,average='micro'),3))+'</td></tr>'
    tr+='<tr><td>8</td><td>Recall Score</td><td>'+str(round(recall_score(y_test,predictions,average='micro'),3))+'</td></tr>'
    tr+='<tr><td>9</td><td>F1 Score</td><td>'+str(round(f1_score(y_test,predictions,average='micro'),3))+'</td></tr>'
    
    # wall-times
    tr+='<tr><td>10</td><td>Wall times</td><td>'+str(walltimes)+' s</td></tr>'
    
    tbl+=th + tr
    tbl+='</table>'
    return(tbl)
    
def printpredicttable(vectorizer,classifier,predictions,test,test_raw_text,y_test,categories,walltimes,isPredicted):
    # isPredicted = True -> use predicted result for evaluating
        
    if isPredicted:
        XXX_test = vectorizer.transform(test)
        start=time.time()
        predictions = classifier.predict(XXX_test)
        walltimes = round(time.time()-start,0)
    
    scores = printscoretable(y_test, predictions, walltimes)
    report = classification_report(y_test, predictions, target_names=categories)
    #predictions_proba = classifier.predict_proba(XXX_test)
    arr_lb = predictions.toarray()
    
    labels = [""] * arr_lb.shape[0]
    for i in range(len(arr_lb)):
        labels[i] = ""
        for j in range(len(arr_lb[i])):
            if (arr_lb[i][j]==1):
                if (labels[i] == ""):
                    labels[i]+=categories[j]
                else:
                    labels[i]+=", "+categories[j]
    if isPredicted:
        text = test_raw_text.iloc[test.index]['Message']
    else:
        text = test
    #text = x_test #.iloc[test.index]['Message'] 
    result = np.column_stack((text,labels))
    #print(result)
    columns = ['Message','Kết quả phân lớp']
    tbl_result = pd.DataFrame(result, columns=columns)
    
    data_in_html = tbl_result.to_html(index=False)

    total_id = 'totalID'
    header_id = 'headerID'
    style_in_html = """<style>
        table#{total_table} {{font-size:13px; text-align:left; border:2px solid red;
                             width:100%}}
        thead#{header_table} {{background-color: #4D4D4D; color:#ffffff;text-align:center}}
        </style>""".format(total_table=total_id, header_table=header_id)
    data_in_html = re.sub(r'<thead', r'<thead id=%s ' % header_id, data_in_html)
    data_in_html = re.sub(r'<td>', r'<td style="text-align:left">', data_in_html)
    
    return [style_in_html+data_in_html, scores, report]

# Cac thu vien mo hinh classification
def BinaryRelevance_GNB(x_train, y_train, x_test):
    classifier_br1 = BinaryRelevance(GaussianNB())

    # train
    classifier_br1.fit(x_train, y_train)

    # predict
    predictions_br1 = classifier_br1.predict(x_test)
    
    return [classifier_br1, predictions_br1]

def BinaryRelevance_SVC(x_train, y_train, x_test):
    classifier_br2 = BinaryRelevance(
        classifier = SVC(kernel='linear'),
        require_dense = [False, True]
    )
    # train
    classifier_br2.fit(x_train, y_train)

    # predict
    predictions_br2 = classifier_br2.predict(x_test)
    
    return [classifier_br2, predictions_br2]

def BinaryRelevance_LR(x_train, y_train, x_test):
    classifier_br3 = BinaryRelevance(LogisticRegression())

    # train
    classifier_br3.fit(x_train, y_train)

    # predict
    predictions_br3 = classifier_br3.predict(x_test)
    
    return [classifier_br3, predictions_br3]

def ClassierChains(x_train, y_train, x_test):
    classifier_cc = ClassifierChain(GaussianNB())

    # Training logistic regression model on train data
    classifier_cc.fit(x_train, y_train)

    # predict
    predictions_cc = classifier_cc.predict(x_test)
    
    return [classifier_cc, predictions_cc]

def MlkNN(x_train, y_train, x_test, _k, _s):
    if _k is None: _k=12;
    if _s is None: _s=1.0;

    classifier_ml = MLkNN(k=_k, s=_s)

    # This classifier can throw up errors when handling sparse matrices.
    x_train = lil_matrix(x_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    x_test = lil_matrix(x_test).toarray()

    # train
    classifier_ml.fit(x_train, y_train)

    # predict
    predictions_ml = classifier_ml.predict(x_test)
    
    return [classifier_ml, predictions_ml]