#######################################################################################
#Importing required libraries
#######################################################################################

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
nltk.download()

########################################################################################
#Reading data and perform basic analysis and visulaization to understand the data
########################################################################################

#read the data with required columns
fields=['TRANS_CONV_TEXT','Patient_Tag']
textconv=pd.read_csv("C:/Users/jabhishe/Desktop/Online Hackathons/Hack/train.csv",encoding ='latin1',usecols=fields)
textconv['TRANS_CONV_TEXT'] = textconv['TRANS_CONV_TEXT'].values.astype(str)
textconv.head()

#read test data
field=['Index','TRANS_CONV_TEXT']
test=pd.read_csv("C:/Users/jabhishe/Desktop/Online Hackathons/Hack/test.csv",encoding ='latin1',usecols=field)
test['TRANS_CONV_TEXT'] = test['TRANS_CONV_TEXT'].values.astype(str)

#Exploratory Data Analysis
textconv.describe()
textconv.groupby('Patient_Tag').describe()

#Detecting the lngth of the messages by visualization
textconv['length']=textconv['TRANS_CONV_TEXT'].str.len()
textconv['length'].plot(bins=50,kind='hist')#visualizing the frequency of words

###########################################################################################
#Text Pre-Processing
###########################################################################################

wnl=WordNetLemmatizer()

#Defining function to remove numbers, punctuation and stop words
def text_process(text):
    text=re.sub(r'\d+','',text)#removing numbers
    text=wnl.lemmatize(text)#lemmitization
    nopunc =[i for i in text if i not in string.punctuation]#removing punctuations
    nopunc=''.join(nopunc)
    return [w for w in nopunc.split() if w.lower() not in stopwords.words('english')]#removing stopwords

#check the function
textconv['TRANS_CONV_TEXT'].head(5).apply(text_process)

#Converting lists of tokens into vectors
bag_t=CountVectorizer(analyzer=text_process).fit(textconv['TRANS_CONV_TEXT'])
print(len(bag_t.vocabulary_))

#applying transform on entire bag of words
conv_b=bag_t.transform(textconv['TRANS_CONV_TEXT'])

#Weighting and Normalization (using TfidfTransformer)
tfidf_transform=TfidfTransformer().fit(conv_b)
tfidf_conv=tfidf_transform.transform(conv_b)
print("Shape: ",tfidf_conv.shape)
#Shape:  (1157, 29435)

#training the model
classifier=LogisticRegression()
patient_detect_model=classifier.fit(tfidf_conv,textconv['Patient_Tag'])

#Model Evaluation
predict_all = patient_detect_model.predict(tfidf_conv)
print(set(predict_all))

print(classification_report(textconv['Patient_Tag'],predict_all))
print(confusion_matrix(textconv['Patient_Tag'],predict_all))

#############################################################################################
#Modelling by Dividing the train and validation data 
#############################################################################################

#Test_train Split
X_train,X_valid,y_train,y_valid = train_test_split(textconv['TRANS_CONV_TEXT'],textconv['Patient_Tag'],test_size=0.3)

#creating a pipeline

pipeline = Pipeline([
   ( 'bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',LogisticRegression()),
])

#Model Fit
model=pipeline.fit(X_train,y_train)

#Prediction for validation data
y_pred=model.predict(X_valid)

################################################################################################
#Model evaluation
################################################################################################

accuracy_score(y_valid, y_pred)
#accuracy : 85.05%
print(classification_report(y_valid,y_pred))
#Confusion matrix
print(confusion_matrix(y_valid,y_pred))

################################################################################################
#Prediction
################################################################################################
#Predict the test data
y_test_predict=model.predict(test['TRANS_CONV_TEXT'])
#Including predicted Patient Tag column in the test data
test['Patient_Tag']=y_test_predict
test=test[["Index","Patient_Tag"]]
#Importing predicted Patient tag to excel
export_excel = test.to_csv (r"C:/Users/ripzs/Desktop/Hackathon/datasetc062cf9/dataset/Test_pred.csv", index = None, header=True) 
