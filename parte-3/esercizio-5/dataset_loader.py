import os, random
import numpy as np
from nltk.tokenize import word_tokenize
from embeddings import word_vectors
from utils import *

def load_categories(dataset_path):
    subfolders = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
    category_idx = {name: i for i, name in enumerate(sorted(subfolders))}
    return category_idx

def load_dataset(dataset_path, category_idx):
    X = []
    y = []

    for category, idx in category_idx.items():
        categ_folder = os.path.join(dataset_path, category)

        for article_text in read_all_articles_in_folder(categ_folder):
            words = word_tokenize(article_text)
            words = [word.lower() for word in words]

            vectorized_words = []

            for word in words:
                try:
                    vec = word_vectors[word]
                    vectorized_words.append(vec)
                except KeyError:
                    pass

            X.append(np.array(vectorized_words))

            categories = [0] * len(category_idx)
            categories[category_idx[category]] = 1
            y.append(np.array(categories))

    return X, y

def pad(data, length, vec_len):
    new_data = []
    for x in data:
        if len(x) == length:
            new_data.append(x)
        elif len(x) < length:
            x = list(x)
            while(len(x) < length):
                x.append(np.zeros(vec_len, dtype=np.float32))
            new_data.append(np.array(x))
        else:
            x = list(x)
            new_x = []
            i = 0
            while(len(new_x) < length):
                new_x.append(x[i])
                i += 1
            new_data.append(np.array(new_x))
    return new_data

def randomize_dataset(X, y):
    tmp = list(zip(X, y))
    random.shuffle(tmp)
    X, y = zip(*tmp)
    return X, y

def preprocess_sentence(text, max_len, vec_len):
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    vectorized_words = []
    X = []
    for word in words:
        try:
            vec = word_vectors[word]
            vectorized_words.append(vec)
        except KeyError:
            pass

    X.append(np.array(vectorized_words))
    return pad(X, max_len, vec_len)
