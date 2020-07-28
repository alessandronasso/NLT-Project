import pandas as pd
import csv, re
import numpy as np
from itertools import product, starmap
from functools import partial

DIM = 300

def load_nasari(path, dim=DIM):
    # Metodo che si occupa di caricare il file NASARI
    nasari = pd.read_csv(path, delimiter='\t', quoting=csv.QUOTE_NONE, names=['BabelSynsetID'] + list(range(dim)))
    # Qui avviene lo split della prima cella che identifica il BabelSynsetID ed il suo nome
    # in due variabili diverse
    new_df = nasari.loc[:,'BabelSynsetID'].str.split('__', expand=True)
    nasari['BabelSynsetID'] = new_df[0]
    nasari['Name'] = new_df[1].str.lower().str.replace('_', ' ')
    # Ai due campi precedentemente definiti accodo i restanti valori
    nasari = nasari[['BabelSynsetID', 'Name'] + list(range(dim))]
    # Rimuovo testo all'interno di parentesi per esempio "Play (Theatre)" -> "Play"
    nasari['Name'] = nasari['Name'].apply(lambda x: re.sub('(.*?)\(.*?\)', '\g<1>', x).strip())
    return nasari

def load_synset_ids(path, nasari):
    # Metodo per creare un dizionario in cui associo una parola
    # ad un dizionario che a sua volta associa i BabelSynsetID di quella parola
    # al vettore nasari corrispondente
    word2synset = {}
    with open(path, 'r', encoding='UTF-8') as f:
        cur_key = ''
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#'):
                cur_key = line[1:]
                word2synset[cur_key] = {}
            else:
                vec = nasari.loc[nasari['BabelSynsetID'] == line]
                if not vec.empty:
                    word2synset[cur_key][line] = vec
                else:
                    word2synset[cur_key][line] = None
    return word2synset

# word_pairs corrisponde all'indice del dataframe con le coppie di parole
# word2synset è il dizionario chiave = termine, valore = dizionario synset-vettore
def get_all_labels(word_pairs, word2synset):
    labels_parola_1 = set()
    labels_parola_2 = set()
    parola_1 = set(word_pairs[i][0] for i in range(len(word_pairs)))
    parola_2 = set(word_pairs[i][1] for i in range(len(word_pairs)))
    for k, v in word2synset.items():
        if k in parola_1:
            labels_parola_1 = labels_parola_1.union(v)
        if k in parola_2:
            labels_parola_2 = labels_parola_2.union(v)
    return (labels_parola_1, labels_parola_2)

def find_max_similarity(vectors_a, vectors_b, similarity_function):
    # Trova la similarità massima tra il prodotto cartesiano di tutti
    # i vettori
    return max(starmap(similarity_function, product(vectors_a, vectors_b)), default=0)

def find_most_similar_synsets(vectors_a, vectors_b, similarity_function, max_similarity):
    for s1, s2 in product(vectors_a, vectors_b):
        # Applico la cosine_similarity su tutte le coppie di vettori
        sim = cosine_similarity(s1, s2)
        if sim == max_similarity:
            bs1 = s1['BabelSynsetID'].iloc[0] if not s1 is None else '-NA-'
            bs2 = s2['BabelSynsetID'].iloc[0] if not s2 is None else '-NA-'
            return (bs1, bs2)

def norm(v):
    # Funzione che calcola la lunghezza euclidea di un vettore
    return np.sqrt(np.sum(abs(v)**2))

def cosine_similarity(v1, v2, dim=DIM):
    if v1 is None or v2 is None:
        return 0
    v1 = v1.loc[:,range(dim)].values[0]
    v2 = v2.loc[:,range(dim)].values[0]
    numeratore = np.dot(v1, v2)
    norma1 = norm(v1)
    norma2 = norm(v2)
    denominatore = norma1 * norma2
    return numeratore / denominatore

max_cosine_similarity = partial(find_max_similarity, similarity_function=cosine_similarity)
