import pandas as pd
import csv
import os, hashlib, pickle
from functools import lru_cache
from time import time
from contextlib import contextmanager

@contextmanager
def timing(message='Elapsed time:', tail_message='secondi'):
    # context manager per calcolare il tempo di esecuzione di un blocco di codice
    start = time()
    yield
    elapsed_time = time() - start
    print(f"{message} {elapsed_time} {tail_message}")

class cached_property(object):
    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def ask(question, options=None, validator=None):
    answer = ''
    if options != None:
        while not answer in options:
            answer = input(question)
    else:
        while not validator(answer):
            answer = input(question)
    return answer

@lru_cache(maxsize=30)
def w(i):
    # ottengo indice per la colonna del dataframe pandas che contiene i pesi della parola i-esima
    return str(i) + 'w'

def vec_to_dict(v, dim):
    # trasformo vettore nasari in dizionario con:
    # chiave = parola
    # valore = rank della parola
    d = {}
    for i in range(dim):
        if v[i] != '-NA-':
            d[v[i]] = i + 1
    return d

def vec_to_inverse_rank_dict(v, dim):
    # trasformo vettore nasari in dizionario con:
    # chiave = parola
    # valore = peso della parola
    d = {}
    for i in range(dim):
        if v[i] != '-NA-':
            d[v[i]] = dim - i
    return d

def weighted_overlap(v1, v2, dim):
    # misura di similarità tra vettori
    d_v1 = vec_to_dict(v1, dim)
    d_v2 = vec_to_dict(v2, dim)
    overlap = set(d_v1.keys()) & set(d_v2.keys())
    s = 0
    d = 0
    idx = 1
    if len(overlap) == 0:
        return 0
    for word in overlap:
        s += 1 / (d_v1[word] + d_v2[word])
        d += 1 / (2 * idx)
        idx += 1
    return s / d

def load_stopwords(path):
    if path == None: return None
    with open(path, 'r', encoding="utf-8") as f:
        words = [line[:-1] for line in f.readlines()]
    return set(words)

def file_hash(path, BUF_SIZE = 65536):
    # ritorno l'hash SHA1 di un file
    # usato per dare un nome al file preprocessato di nasari da salvare su disco
    # per questioni di performance (caching del dataset preprocessato)
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

def load_nasari(path, dim, chunksize, use_pickle):
    # parametri
    # path = percorso del file nasari
    # dim = numero di dimensioni (colonne) da leggere
    # chunksize = il file viene letto a blocchi di tale dimensione (risparmio RAM)
    # use_pickle = True per salvare/caricare i dati preprocessati
    if use_pickle:
        # calcolo il path in cui salvare / da cui caricare i dati nasari preprocessati
        original_hash = file_hash(path)
        absp = os.path.abspath(path)
        dirn = os.path.dirname(absp)
        pickle_path = os.path.join(dirn, f'nasari.{original_hash}.{dim}.pkl')
    
    if use_pickle and os.path.isfile(pickle_path):
        # se i dati preprocessati sono già salvati su disco li carico e li ritorno
        with open(pickle_path, 'rb') as f:
            nasari = pickle.load(f)
        return nasari
    
    # altrimenti, carico il file originale di nasari e lo preprocesso
    columns = ['BabelSynsetID', 'Name'] + list(range(dim)) + [str(i)+'w' for i in range(dim)]
    nasari = pd.DataFrame(columns=columns)
    columns = ['BabelSynsetID', 'Name'] + list(range(dim))
    
    for chunk in pd.read_csv(path, sep=';', quoting=csv.QUOTE_NONE, names=columns, chunksize=chunksize, low_memory=False):
        chunk.fillna('-NA-_0', inplace=True)
        for i in range(dim):
            chunk[[i, str(i)+'w']] = chunk.loc[:,i].str.split('_',expand=True)
            chunk[str(i)+'w'] = chunk[str(i)+'w'].astype(float)
        
        chunk.loc[:,'Name'] = chunk.loc[:,'Name'].str.lower()
        nasari = nasari.append(chunk)
    
    nasari = nasari.set_index('Name').sort_index()
    if use_pickle:
        # salvo i dati preprocessati su disco
        with open(pickle_path, 'wb') as f:
            pickle.dump(nasari, f)
        
    return nasari

from summarization import Document
def load_document(path, nasari, options_dict={}):
    # metodo per caricare il documento da riassumere
    with open(path, 'r', encoding="utf-8") as f:
        text = ''.join(f.readlines())
        doc = Document(text, nasari, **options_dict)
    return doc
