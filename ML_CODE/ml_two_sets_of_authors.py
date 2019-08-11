import numpy as np
import mysql.connector as sql
import pandas as pd
import re
import nltk
import textstat
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

#procedure to find the syllable count
def syllable_count(word):
    word = word.lower()
    scount = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        scount += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            scount += 1
    if word.endswith("e"):
        scount -= 1
    if scount == 0:
        scount += 1
    return scount

#obtaining the list of all authors
file=open('1_1000.txt','r')
a=file.readlines()
for i in range(len(a)):
    a[i]=re.sub('\n$','',a[i])
file.close()

#obtaining the list of authors in the first set 
ab=[]
file=open('1_500.txt','r') 
ab=file.readlines()
for i in range(len(ab)):
    ab[i]=re.sub('\n$','',ab[i])
file.close()

#obtaining the list of authors in the second set 
ac=[]
file=open('501_1000.txt','r') 
ac=file.readlines()
for i in range(len(ac)):
    ac[i]=re.sub('\n$','',ac[i])
file.close()



accuracy=0
precision=0
recall=0
fscore=0

# generating a dictionary for avergae word frequency class
file1 = open("avg_wrd_freq.txt","r") 

d={}
inp=file1.readlines()
for i in range(len(inp)):
    inp[i]=re.sub(r'\n$','',inp[i])
    (w,c)=inp[i].split()
    d[w]=c
file1.close()

for count in range(len(a)):
    #obtaining positive class reviews
    connection = sql.connect(host='localhost', database='amazon_db_software', user='Aparna', password='anu90appu96@')
    cursor = connection.cursor()
    stt = 'select * from reviews where customer_id="'+a[count]+'"  limit 50' 
    cursor.execute(stt)
    rows = cursor.fetchall()
    df0=pd.DataFrame(rows)
    df0=df0[:50]
    df0[14]="yes"
    
    # obtaining negative class reviews   
    if a[count] in ab:            
        stt = 'select * from reviews where customer_id in '+ str(tuple(ac))+'  limit ' + str(len(df0))
    else:
        stt = 'select * from reviews where customer_id in '+ str(tuple(ab))+'  limit ' + str(len(df0)) 
    cursor.execute(stt)
    rows = cursor.fetchall()
    df1=pd.DataFrame(rows)
    df1=df1[:50]
    df1[14]="no"    
    dff=pd.DataFrame(np.concatenate((df0,df1)))
    reviews=dff.iloc[:,10]
    reviews=reviews.replace(r'\n','',regex=True)
    
    stop_words=set(stopwords.words('english'))
    Y=dff.iloc[:,14]
    
    #removing hyperlinks
    reviews=reviews.replace(r"http\S+|www\S+"," ",regex=True)
    
    #initializing
    corpus=[]
    stopcount=[]
    kingrade=[]
    gunning=[]
    postag=[]
    avg_words_per_sent=[]
    avg_char_count=[]
    avg_syl_cnt=[]
    final_pos=[]
    pos_trigram_diversity=[]
    punct_count=[]
    difficult_words1=[]
    flesch_reading_ease1=[]
    smog_index1=[]
    automated_readability_index1=[]
    coleman_liau_index1=[]
    linsear_write_formula1=[]
    dale_chall_readability_score1=[]
    avg_word_freq=[]
    capital_count=[]
    
    for i in range(len(reviews)):
        #obtaining capital word count
        review_captial=reviews[i].strip().replace("\'",'')
        cnt = 0
        cap_words = word_tokenize(review_captial)
        cap_words=[w for w in cap_words if w not in ['.',',',';','?',':','!','"',"'",'#']]
        for w in cap_words:
            if w[0].isupper():                
                cnt += 1      
        capital_count.append(cnt/len(cap_words))
                    
        #obatining readability features
        reviews[i]=reviews[i].strip().lower().replace("\'",'')
        kingrade.append(textstat.flesch_kincaid_grade(reviews[i]))
        gunning.append(textstat.gunning_fog(reviews[i]))         
        flesch_reading_ease1.append(textstat.flesch_reading_ease(reviews[i]))
        difficult_words1.append(textstat.difficult_words(reviews[i]))
        smog_index1.append(textstat.smog_index(reviews[i]))
        automated_readability_index1.append(textstat.automated_readability_index(reviews[i]))
        coleman_liau_index1.append(textstat.coleman_liau_index(reviews[i]))
        linsear_write_formula1.append(textstat.linsear_write_formula(reviews[i]))
        dale_chall_readability_score1.append(textstat.dale_chall_readability_score(reviews[i])) 
        word_freq=[]
        
        #obtaining punctuation count
        words = word_tokenize(reviews[i])        
        punct=[w for w in words if w in ['.',',',';','?',':','!']]
        punct_count.append(len(punct)/len(words))
        word=[w for w in words if w not in ['.',',',';','?',':','!','"',"'",'#']]
        corpus.append(reviews[i])
        
        #obtaining stopwords frequency
        stop=[w for w in word if w in stop_words]
        stop_freq=len(stop)/len(word)
        stopcount.append(stop_freq)
        
        #average words per sentence
        sentences = nltk.sent_tokenize(reviews[i])
        avg_words_count=float(len(word))/len(sentences)
        avg_words_per_sent.append(avg_words_count)
        
        # average word frequency class, assiging a value of 10 (mean score) to the words not in the dictionary
        count=0
        syl_cnt = 0
        for w in words:
            if w in ['.',',',';','?',':','!','"',"'",'"',"'",'#']:
                continue
            if w in d.keys():
                word_freq.append(d[w])
            else:
                word_freq.append(10) 
        ll=0
        # average word frequency
        for l in range(len(word_freq)):
            ll+=float(word_freq[l])
        avg_word_freq.append(ll/len(word_freq))
        
        # average character and syllable count    
        for w in word:
            count+=len(w)
            syl_cnt += syllable_count(w)
        avg_char_count.append(float(count)/len(word))
        avg_syl_cnt.append(float(syl_cnt) / float(len(word)))
        
        # pos trigramdiversity
        pos=""
        postag.append(nltk.pos_tag(word))      
        for j in range(len(postag[i])):
            pos+=str(postag[i][j][1])+" "
        final_pos.append(pos)
        
        vectorizer_pos = CountVectorizer(stop_words=None,analyzer='word',ngram_range=(3,3),token_pattern = r"(?u)\b\w+\b")
        pos_tri_diversity = vectorizer_pos.fit_transform([pos]).toarray()
        c=0.0
        for i in range(len(pos_tri_diversity[0])):
            if pos_tri_diversity[0][i]==1:
                c+=1
        pos_trigram_diversity.append(c/len(pos_tri_diversity[0]))

    stopcount=np.float_(stopcount)
    #word unigram
    vectorizer = TfidfVectorizer()
    word_uni = vectorizer.fit_transform(corpus).toarray()
    
    #character unigram, bigram and trigram
    vectorizer1 = TfidfVectorizer(analyzer='char',ngram_range=(1,1))
    char_uni = vectorizer1.fit_transform(corpus).toarray()
    vectorizer2 = TfidfVectorizer(analyzer='char',ngram_range=(2,2))
    char_bi = vectorizer2.fit_transform(corpus).toarray()
    vectorizer3 = TfidfVectorizer(analyzer='char',ngram_range=(3,3))
    char_tri = vectorizer3.fit_transform(corpus).toarray()
    
    #pos unigram, bigram and trigram
    vectorizer33 = CountVectorizer(stop_words=None,analyzer='word',ngram_range=(3,3))
    pos_tri = vectorizer33.fit_transform(final_pos).toarray()
    vectorizer22 = CountVectorizer(stop_words=None,analyzer='word',ngram_range=(2,2))
    pos_bi = vectorizer22.fit_transform(final_pos).toarray()
    vectorizer11 = CountVectorizer(stop_words=None,analyzer='word',ngram_range=(1,1))
    pos_uni = vectorizer11.fit_transform(final_pos).toarray()
 
    X=np.column_stack((word_uni,char_uni,char_bi,char_tri,stopcount,kingrade,gunning,avg_words_per_sent,avg_char_count,avg_syl_cnt,pos_uni,pos_bi,pos_tri,pos_trigram_diversity,punct_count,flesch_reading_ease1,difficult_words1,smog_index1,automated_readability_index1,coleman_liau_index1,linsear_write_formula1,dale_chall_readability_score1,avg_word_freq,capital_count))
      
    #normalizing data between 0 to 1
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler(copy=True, feature_range=(0, 1))
    X=scaler.transform(X)  
  
    labelencoder_Y=LabelEncoder()
    Y=labelencoder_Y.fit_transform(Y)
    
    #SVM with 5 fold cross validation   
    clf = LinearSVC(max_iter=10000)   
    scoring=['accuracy','precision_macro', 'recall_macro','f1_macro']    
    scores=cross_validate(clf, X, Y, scoring=scoring, cv=5, return_train_score=False)
    
    # performance metrics
    accuracy+=scores['test_accuracy'].mean()
    precision+=scores['test_precision_macro'].mean()
    fscore+=scores['test_f1_macro'].mean()
    recall+=scores['test_recall_macro'].mean()
    print(scores['test_accuracy'].mean())
    
# average value for 1000 authors     
print("final average accuracy")
print(accuracy/float(len(a)))
print("final precision")
print(precision/float(len(a)))
print("final recall")
print(recall/float(len(a)))
print("final fscore")
print(fscore/float(len(a)))






