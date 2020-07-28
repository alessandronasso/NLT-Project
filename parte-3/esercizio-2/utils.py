import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from statistics import mean
from itertools import product, starmap
from functools import lru_cache, partial
from nltk.corpus import wordnet as wn
from collections import Counter

additional_stopwords = ['\'s', '’']
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    sentence = word_tokenize(sentence)
    stopwordset = set(stopwords.words('english') + additional_stopwords)
    sentence = {lemmatizer.lemmatize(x.lower()) for x in sentence if not x.lower() in stopwordset and not x in punctuation}
    sentence = {x for x in sentence if len(wn.synsets(x)) > 0}
    return sentence

def word_freq_from_sentence(sentence):
    sentence = word_tokenize(sentence)
    stopwordset = set(stopwords.words('english') + additional_stopwords)
    sentence = [lemmatizer.lemmatize(x.lower()) for x in sentence if not x.lower() in stopwordset and not x in punctuation]
    return Counter(sentence)

def union(row):
    res = set()
    for s in row.values:
        res = res.union(s)
    return frozenset(res)

def counter(row):
    res = Counter()
    for s in row.values:
        res += Counter(s)
    return res

@lru_cache(maxsize=256)
def get_signature(sense, expand=True):
    # Per ottenere la signature splitto la frase in singole 
    # parole tramite word_tokenize e poi estraggo il contesto
    signature = preprocess(sense.definition())
    
    # Effettuo la stessa operazione per gli esempi del sysnet 
    # presenti in wordnet
    for example in sense.examples():
        signature = signature.union(preprocess(example))
    
    # Aggiungo i sinonimi lemmatizzati
    synonyms = {lemmatizer.lemmatize(x.lower()) for x in sense.lemma_names() if not '_' in x}
    signature = signature.union(synonyms)

    # Per migliorare la precisione dell'algoritmo aggiungo
    # inoltre gli iperonimi, gli iponimi e i meronimi alla
    # signature
    if expand:
        hypernym_sig, hyponym_sig, meronym_sig = set(), set(), set()
        
        for hypernym in sense.hypernyms() + sense.instance_hypernyms():
            hypernym_sig = hypernym_sig.union(get_signature(hypernym, False))
            
        for hyponym in sense.hyponyms() + sense.instance_hyponyms():
            hyponym_sig = hyponym_sig.union(get_signature(hyponym, False))
            
        for meronym in sense.part_meronyms() + sense.member_meronyms():
            meronym_sig = meronym_sig.union(get_signature(meronym, False))
        
        signature = signature.union(hypernym_sig)
        signature = signature.union(hyponym_sig)
        signature = signature.union(meronym_sig)
    
    return signature

@lru_cache(maxsize=64)
def lesk_disambiguate(context, word, expand=True):
    synsets = wn.synsets(word)
    # Il senso più frequente corrisponde al primo della lista
    best_sense = synsets[0]
    max_overlap = 0
    
    for sense in synsets:
        # Ottengo la signature del synset che sto esaminando
        signature = get_signature(sense, expand)
        # Confronto quanti elementi ci sono in comune tra la signature del
        # senso che sto esaminando e il contesto (la frase)
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense

def disambiguate_all(context, words, expand=True):
    synsets = set()
    for w in words:
        synset = lesk_disambiguate(context, w, expand)
        synsets = synsets.union((synset,))
    return synsets

def wu_palmer(s1, s2):
    sim = s1.wup_similarity(s2)
    return sim if sim is not None else 0

def path_similarity(s1, s2):
    sim = s1.path_similarity(s2)
    return sim if sim is not None else 0

@lru_cache(maxsize=256)
def sense_sig_with_freq(sense, expand=True):
    sig = word_freq_from_sentence(sense.definition())
    for ex in sense.examples():
        sig = sig + word_freq_from_sentence(ex)
    synonyms = {lemmatizer.lemmatize(x.lower()) for x in sense.lemma_names() if not '_' in x}
    sig += Counter(synonyms)
    
    if expand:
        hypernym_sig, hyponym_sig, meronym_sig = Counter(), Counter(), Counter()
        
        for hypernym in sense.hypernyms() + sense.instance_hypernyms():
            hypernym_sig += sense_sig_with_freq(hypernym, False)
            
        for hyponym in sense.hyponyms() + sense.instance_hyponyms():
            hyponym_sig += sense_sig_with_freq(hyponym, False)
            
        for meronym in sense.part_meronyms() + sense.member_meronyms():
            meronym_sig += sense_sig_with_freq(meronym, False)
        
        sig += hypernym_sig
        sig += hyponym_sig
        sig += meronym_sig
    
    return sig
    
def calc_overlap(sense, context):
    sig = sense_sig_with_freq(sense)
    overlap = sig & context
    overlap = sum(overlap.values())
    return overlap
    

def find_most_similar_hyponym(genus, context, max_depth, first=True):
    if first:
        best_sense = wn.synsets('entity')[0]
        max_similarity = 0
    else:
        best_sense = genus
        max_similarity = calc_overlap(genus, context)
    if max_depth == 0:
        return best_sense, max_similarity
    for hypo in genus.hyponyms():
        sense, similarity = find_most_similar_hyponym(hypo, context, max_depth - 1, False)
        if max_similarity < similarity:
            max_similarity = similarity
            best_sense = sense
    return best_sense, max_similarity

# n = numero massimo di elementi presi dalla cima della classifica delle frequenze dei sensi
def wsi(definition_sense_counts, context, max_depth=8, n=20):
    best_sense = wn.synsets('entity')[0]
    max_similarity = 0
    for genus, _ in definition_sense_counts.most_common(n):
        if genus.pos() == 'n':
            sense, similarity = find_most_similar_hyponym(genus, context, max_depth)
            if max_similarity < similarity and sense not in definition_sense_counts.keys():
                max_similarity = similarity
                best_sense = sense
    return best_sense, max_similarity

def load_and_preprocess(path):
    dataset = pd.read_excel(path)
    dataset = dataset.set_index('Concetto')
    dataset = dataset.applymap(preprocess)
    dataset['word_freq'] = dataset.apply(counter, axis=1)
    dataset['all_words'] = dataset.loc[:, dataset.columns != 'word_freq'].apply(union, axis=1)

    # Disambiguazione
    for index, row in dataset.iterrows():
        context = row['all_words']
        
        for col in dataset.drop('all_words', axis=1).drop('word_freq', axis=1):
            definizione = dataset.loc[index, col]
            synsets = disambiguate_all(context, definizione)
            dataset.loc[index, col] = synsets

    dataset['sense_counts'] = dataset.drop('all_words', axis=1).drop('word_freq', axis=1).apply(counter, axis=1)

    return dataset