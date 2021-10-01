from flask import Flask, render_template, request,url_for

# nlp library
import spacy
import pandas as pd
import joblib
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import en_core_web_sm
import string
import dill
import pickle
from textblob import TextBlob
from gensim.summarization import summarize

# load the spacy english model
nlp = en_core_web_sm.load()
punct = string.punctuation
stopwords = list(STOP_WORDS)

# model load spam classification
with open('models/spam.pickle', 'rb') as handle:
	spam_clf = pickle.load(handle)

# news classifier
with open('models/model.pickle', 'rb') as handle:
	model = pickle.load(handle)

app = Flask(__name__)


# home page
@app.route('/')
def index():
	return render_template('home.html')


# sentiment analysis
@app.route('/nlpsentiment')
def sentiment_nlp():
	return render_template('sentiment.html')


@app.route('/sentiment',methods = ['POST','GET'])
def sentiment():
	if request.method == 'POST':
		message = request.form['message']
		# Machine learning analysiser
		blob1 = TextBlob(message)
		pred = blob1.sentiment.polarity
		pred = round(pred)
		return render_template('sentiment.html', prediction=int(pred))

@app.route('/newsclf')
def news_classifier():
	return render_template('news.html')

@app.route('/newsclassifier',methods=['POST','GET'])
def news_clf():
	if request.method == 'POST':
		message = request.form['message']
		pred = model.predict([message])
		return render_template('news.html', prediction=(pred[0]))

# spam
@app.route('/nlpspam')
def spam_nlp():
	return render_template('spam.html')

# spam classification
@app.route('/spam',methods= ['POST','GET'])
def spam():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	#
    X = df['message']
    y = df['class']
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('spam.html', prediction=my_prediction)

# summarize
@app.route('/nlpsummarize')
def summarize_nlp():
	return render_template('summarize.html')

# spam classification
@app.route('/summarize',methods= ['POST','GET'])
def sum_route():
	if request.method == 'POST':
		message = request.form['message']
		sum_message = summarize(message)
		return render_template('summarize.html',original = message, prediction=sum_message)


if __name__ == '__main__':
	app.run(debug=True)