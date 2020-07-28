from utils import *
from tqdm import tqdm
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# DATASET URL
# http://mlg.ucd.ie/datasets/bbc.html

VERB = 'buy'
OUT_FILE = 'buy.5000.txt'

SENTENCES_FILE_PATHS = 'europarl/europarl-v7.it-en.en'
MAX_SENTENCES = 5000

if __name__ == '__main__':
    sentences = read_sentences(SENTENCES_FILE_PATHS)
    sentence_count = len(sentences)

    selected_sentences = []
    selected_count = 0

    lemmatizer = WordNetLemmatizer()

    for text in tqdm(sentences, total=sentence_count, unit=' sentences'):
        words = word_tokenize(text)
        lemmatized = set(lemmatizer.lemmatize(x) for x in words)
        if VERB in lemmatized:
            selected_sentences.append(text)
            selected_count += 1
        if selected_count >= MAX_SENTENCES:
            break

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(selected_sentences)
