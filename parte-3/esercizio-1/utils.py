from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from statistics import mean
from itertools import combinations, starmap

additional_stopwords = ['\'s', '’']
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    sentence = word_tokenize(sentence)
    stopwordset = set(stopwords.words('english') + additional_stopwords)
    sentence = {lemmatizer.lemmatize(x.lower()) for x in sentence if not x.lower() in stopwordset and not x in punctuation}
    return sentence

def similarity(s1, s2):
    overlap = len(s1 & s2)
    l = max(len(s1), len(s2))
    return overlap / l

def calc_mean_similarity(frasi, similarity_function):
    # Ritorna la massima similarità tra tutte le coppie di frasi
    return mean(starmap(similarity_function, combinations(frasi, r=2)))
