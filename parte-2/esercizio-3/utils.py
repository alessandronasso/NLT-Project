import csv
from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn

def load_stopwords(path):
    # Metodo che permette di caricare il file contenente
    # le stopwords
    if path == None: return None
    with open(path, 'r', encoding="utf-8") as f:
        words = [line[:-1] for line in f.readlines()]
    return set(words)

def read_annotations(path):
    # Metodo che permette di leggere le tre colonne
    # contenute nel file con le annotazioni
    annotations = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            frame = fn.frame_by_name(row[0].strip())
            parola = row[1].strip()
            synset = row[2].strip()
            if synset == 'None':
                synset = None
            else:
                synset = wn.synset(synset)
            annotations.append((frame, parola, synset))
    return annotations