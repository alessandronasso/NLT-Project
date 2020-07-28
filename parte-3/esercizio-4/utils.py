from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from statistics import mean
from itertools import starmap, combinations
from scipy.spatial.distance import cosine
from math import floor
import numpy as np
import string

lemmatizer = WordNetLemmatizer()

def read_article(path):
    with open(path, 'r') as f:
        article = f.read()
        paragraphs = article.split('\n')
        paragraphs = [x for x in paragraphs if x.strip() != ''][1:]

    sentences = sent_tokenize(article)[1:]
    return sentences, paragraphs

def load_stopwords(path):
    # Metodo che permette di caricare il file contenente
    # le stopwords
    if path == None: return None
    with open(path, 'r', encoding="utf-8") as f:
        words = [line[:-1] for line in f.readlines()]
    return set(words)

def ctx_counter(sentence, stopwordset=None):
    # Splitto le singole parole della frase
    sentence = word_tokenize(sentence)
    # Specifico gli elementi problematici che dovrò
    # escludere in seguito
    blacklist = ["''", "``"] + list(string.punctuation)
    
    if stopwordset == None:
        stopwordset = set(stopwords.words('english'))
    
    # Lemmatizzo tutte le parole che non sono nella blacklist e che non
    # hanno una forma numerica o particolare
    context = [lemmatizer.lemmatize(x.lower()) for x in sentence if not x.lower() in stopwordset and not x in blacklist and not x.isnumeric() and x != '']
    context = [x[1:] if x.startswith('\'') else x for x in context]
    context = [x for x in context if not x in stopwordset and x != '']
    return Counter(context)

def ctx(sentence, stopwordset=None):
    # Splitto le singole parole della frase
    sentence = word_tokenize(sentence)
    # Specifico gli elementi problematici che dovrò
    # escludere in seguito
    blacklist = ["''", "``"] + list(string.punctuation)
    
    if stopwordset == None:
        stopwordset = set(stopwords.words('english'))
    
    # Lemmatizzo tutte le parole che non sono nella blacklist e che non
    # hanno una forma numerica o particolare
    context = {lemmatizer.lemmatize(x.lower()) for x in sentence if not x.lower() in stopwordset and not x in blacklist and not x.isnumeric() and x != ''}
    context = {x[1:] if x.startswith('\'') else x for x in context}
    context = {x for x in context if not x in stopwordset and x != ''}
    return context


def get_signature(sense, stopwordset=None, expand=True):
    signature = ctx(sense.definition(), stopwordset)
    
    # Effettuo la stessa operazione per gli esempi del sysnet 
    # presenti in wordnet
    for example in sense.examples():
        signature = signature.union(ctx(example, stopwordset))
    
    # Aggiungo i sinonimi lemmatizzati
    synonyms = {lemmatizer.lemmatize(x.lower()) for x in sense.lemma_names() if not '_' in x}
    signature = signature.union(synonyms)

    # Per migliorare la precisione dell'algoritmo aggiungo
    # inoltre gli iperonimi, gli iponimi e i meronimi alla
    # signature
    if expand:
        hypernym_sig, hyponym_sig, meronym_sig = set(), set(), set()
        
        for hypernym in sense.hypernyms() + sense.instance_hypernyms():
            hypernym_sig = hypernym_sig.union(get_signature(hypernym, stopwordset, False))
            
        for hyponym in sense.hyponyms() + sense.instance_hyponyms():
            hyponym_sig = hyponym_sig.union(get_signature(hyponym, stopwordset, False))
            
        for meronym in sense.part_meronyms() + sense.member_meronyms():
            meronym_sig = meronym_sig.union(get_signature(meronym, stopwordset, False))
        
        signature = signature.union(hypernym_sig)
        signature = signature.union(hyponym_sig)
        signature = signature.union(meronym_sig)
    
    return signature

def lesk_disambiguate(sentence, word, stopwordset=None, expand=True):
    synsets = wn.synsets(word)
    # Il senso più frequente corrisponde al primo della lista
    best_sense = synsets[0]
    max_overlap = 0
    # Estraggo il contesto della frase
    context = ctx(sentence, stopwordset)
    
    for sense in synsets:
        # Ottengo la signature del synset che sto esaminando
        signature = get_signature(sense, stopwordset, expand)
        # Confronto quanti elementi ci sono in comune tra la signature del
        # senso che sto esaminando e il contesto (la frase)
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense

def preprocess(sentence, stopwordset=None):
    counter = ctx_counter(sentence)
    context = counter.keys()
    context = {x: lesk_disambiguate(sentence, x) for x in context if len(wn.synsets(x)) > 0}
    senses_count = {v: counter[k] for k, v in context.items()}
    return senses_count

def find_avg_similarity(vectors, similarity_function):
    return mean(starmap(similarity_function, combinations(vectors, 2)))

def window_similarity(window, similarity_func=lambda x, y: 1 - cosine(x, y)):
    vectors = [x for x in window.T]
    return find_avg_similarity(vectors, similarity_func)

def preprocess_sentences(sentences, stopwords=None):
    sentence_dict = {}
    all_senses = set()

    for i, sent in enumerate(sentences):
        senses = preprocess(sent, stopwords)
        all_senses = all_senses.union(senses)
        sentence_dict[i] = (sent, senses)
    
    return sentence_dict, all_senses

def cluster_senses(senses, similarity, threshold):
    similarity_clusters = {}
    cluster_indexes = {}

    i = 0
    for current_sense in sorted(senses):
        added = False
        for sense, cluster in similarity_clusters.items():
            if cluster is None:
                sim = similarity(current_sense, sense)
                if sim is not None and sim >= threshold:
                    similarity_clusters[current_sense] = sense
                    cluster_indexes[current_sense] = cluster_indexes[sense]
                    added = True
                    break
        if not added:
            similarity_clusters[current_sense] = None
            cluster_indexes[current_sense] = i
            i += 1

    return cluster_indexes

def create_topic_matrix(sentence_dict, cluster_indexes):
    cluster_count = len(set(cluster_indexes.values()))
    topic_matrix = np.zeros((cluster_count, len(sentence_dict)), dtype=int)

    for i in range(len(sentence_dict)):
        for sense, count in sentence_dict[i][1].items():
            cluster_index = cluster_indexes[sense]
            topic_matrix[cluster_index, i] += count

    return topic_matrix

def rolling_window_similarity(topic_matrix, window_size):
    similarity_data = []
    for i in range(len(topic_matrix.T) - window_size + 1):
        window = topic_matrix[:, i:i+window_size]
        similarity_data.append(window_similarity(window))
    return similarity_data

def choose_split_indexes(similarity_data, window_size, side_exclusion_zone_size):
    minimas = []
    mean_sim = mean(similarity_data)

    for i in range(len(similarity_data) - 2):
        a = similarity_data[i]
        b = similarity_data[i+1]
        c = similarity_data[i+2]
        if b - a < 0 and c - b > 0 and similarity_data[i+1] < mean_sim:
            if i+1 >= side_exclusion_zone_size and i+1 <= len(similarity_data) - side_exclusion_zone_size:
                minimas.append(i+1)

    split_indexes = set()
    for index in minimas:
        split_indexes.add(index + floor(window_size / 2))

    return split_indexes, minimas
