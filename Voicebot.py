import nltk         
import numpy as np
import random
import string 
import warnings
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os
warnings.filterwarnings('ignore')
import speech_recognition as sr
import pyttsx3
from nltk.stem import WordNetLemmatizer

## We will write a function to classify user input, which uses nps_chat corpora and naive Bayes classifier to categorize the input type by classifying them into listed categories.
# Where else Bot will answer only to Question type classes.
## Greet , Bye , Clarify , Emotion , Emphasis , Reject, Statement, System
## nAnswer, whQuestion, yAnswer, ynQuestion, Other

## below code will set the property of voice engine
e = pyttsx3.init()
voices = e.getProperty('voices')
e.setProperty('voice', voices[0].id)
r = sr.Recognizer()

def takecommand():
    with sr.Microphone() as source:  ## this make the microphone able to listen voice 
        print("Listening...")
        # e.say("to whome you want to send a message ")
        r.adjust_for_ambient_noise(source) # this line adjust your voice and remove the noise
        r.pause_threshold=0.5
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, key=None, language='en-IN') # this line will recognize your voice and and store it in query
        print("You said: ",query,"\n")
    except Exception as obj:
        e.say("Say that again please")
        e.runAndWait()
        takecommand()
        return "none"
    return query

## we read the first 10000 data from nps_chat.xml file

posts = nltk.corpus.nps_chat.xml_posts()[:10000]

## below code will generate a featureset based on the data given by user
## as well as dataset provide by us for training purpose 
def find_features(posts):
    features = {}
    for word in nltk.word_tokenize(posts):
        features['contains({})'.format(word.lower())]= True
    return features

""" in below line (post.text) give you the text and post.get('class') give you the class label
    e.g if post.text = hey everyone 
            print(post.get('class')) -> O/P = Greet 
            list((post.get('word),post.get('pos')) for post in posts[0]) -> O/P = [('hey','UH'),('everyone','NN')]
                                                                        here UH and NN is POS(part of speech) like noun , pronoun etc.
"""

featuresets = [(find_features(post.text),post.get('class')) for post in posts]
# print("features",featuresets)

"""You could train and test on the same dataset, but this would present you with some serious bias issues, 
so you should never train and test against the exact same data. 
To do this, since we've shuffled our data set, 
we'll assign the first featuresets[len(featureset)*0.1:] shuffled reviews, consisting of both positive and negative reviews, as the training set. 
"""

size = int(len(featuresets)*0.1)
train_set,test_set = featuresets[size:],featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

## now print the accuracy of our classifier 
# print("Classifier accuracy percent",(nltk.classify.accuracy(classifier,train_set)*100))

## now creating a greeting function 
GREETING_INPUTS = ("hello","hay","hi","greetings","sup","what's up","hey",)
GREETING_RESPONSES = ["hi","hey","hi there","hello","I am glad! You are talkinig to me"]

def greeting(sentence):
    ## if user input is greeting , return a greeting response 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

## reading in the input_corpus

# with open('question_answer_pairs.txt','r', encoding='utf8',errors='ignore') as fin:
#     raw = fin.read().lower()

with open('F:/machine learning/practical_example/Projects/train.txt','r', encoding='utf8',errors='ignore') as fin:
    raw = fin.read().lower()

## Tokenize the input 
sent_tokens = nltk.sent_tokenize(raw) ## convert to list of sentences
word_tokens = nltk.word_tokenize(raw) ## converts to list to words

## Lemmatization is the more sophisticated than stemming .
## stemmer works on an individual word without knowledge of the context
## for Ex. word "better" has "good" as its lemma

## this below function return the Lemma means (root word) e.g "better"->"good"
lem = WordNetLemmatizer()
def LemTokens(tokens):
    return [lem.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct),None)for punct in string.punctuation) ## this line remove the punctuation from string 
# print("remove_punct_dict:",remove_punct_dict)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

## Generate response and processing
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response) ## append user input at last in sent_token
    """ We initialize the tfidfvectorizer and then convert all the sentences in the 
        corpus along with the input sentence into their corresponding vectorized form."""

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)  ## this line learn vocabulary and idf, return document-term matrix
    # print(TfidfVec.get_feature_names()) 

    """ We use the cosine_similarity function to find the cosine similarity between the last item in the tfidf list 
        (which is actually the word vector for the user input since it was appended at the end) 
        and the word vectors for all the sentences in the corpus."""

    vals = cosine_similarity(tfidf[-1],tfidf)
    """We sort the list containing the cosine similarities of the vectors, 
       the second last item in the list will actually have the highest cosine (after sorting) with the user input. 
       The last item is the user input itself, therefore we did not select that."""
    idx = vals.argsort()[0][-2]
    print('idx',idx)
    flat = vals.flatten()
    print("flat",flat)
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        robo_response = robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = sent_tokens[idx+1].split('\n')[0]
        print("sent_tokens",type(sent_tokens[idx+1]))
        print("response ",robo_response)
        return robo_response


flag = True
e.say(" Welcome to Voice Chatbot ")
e.runAndWait()
## take the input from user 
while(flag):
    
    user_response = takecommand()
    user_txt = user_response
    clas= classifier.classify(find_features(user_response)) ## classify(featureset) function give the most appropriate label for featureset
    if (user_txt!='stop'):
        if(clas=='Emotion'):
            flag = False
            e.say("you are welcome")
            e.runAndWait()
        else:
            if(greeting(user_response)!=None):
                e.say(greeting(user_response))
                e.runAndWait()
            else:
                res = (response(user_response))
                sent_tokens.remove(user_response)
                e.say(res)
                e.runAndWait()
    else:
        flag = False
        e.say("bye! Take care")
        e.say("have a good day")
        e.runAndWait()

print("done")