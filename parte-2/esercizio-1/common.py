import numpy as np
from nltk.corpus import wordnet as wn
from itertools import product, starmap
from functools import lru_cache
from collections import deque

# CONSTANTS
MAX_DEPTH = max(max(len(hyp_path) for hyp_path in ss.hypernym_paths()) for ss in wn.all_synsets())
WORDSIM_PATH = 'WordSim353.csv'


def find_max_similarity(synset_a, synset_b, similarity_function):
    # Ritorna la massima similarità tra tutte le coppie di sensi nei synsets
    return max(starmap(similarity_function, product(synset_a, synset_b)), default=0)


@lru_cache(maxsize=256)
def max_depth(sense):
    # Ottengo la distanza massima dalla root per il senso passato come param.
    hypernyms = sense.hypernyms() + sense.instance_hypernyms()
    
    if len(hypernyms) == 0:
        sense_max_depth = 0
    else:
        # Se non ho raggiunto la root procedo con l'esplorare
        # gli altri iperonimi e cercare il path più lungo
        sense_max_depth = 1 + max(max_depth(x) for x in hypernyms)

    return sense_max_depth


@lru_cache(maxsize=256)
def min_depth(sense):    
    # Ottengo la distanza minima dalla root per il senso passato come param. 
    hypernyms = sense.hypernyms() + sense.instance_hypernyms()
    
    if len(hypernyms) == 0:
        sense_min_depth = 0
    else:
        # Se non ho raggiunto la root procedo con l'esplorare
        # gli altri iperonimi e cercare il path più corto
        sense_min_depth = 1 + min(min_depth(x) for x in hypernyms)
    
    return sense_min_depth


@lru_cache(maxsize=256)
def min_depth_through_node(sense, node):    
    # Calcolo la profondità minima dalla node a sense
    hypernyms = sense.hypernyms() + sense.instance_hypernyms()
    
    if sense == node:
        # Se ho raggiunto l'obiettivo
        sense_min_depth = 0
    elif len(hypernyms) == 0:
        # Sono arrivato alla radice seguendo un path che non passa per node
        sense_min_depth = 1000000
    else:
        # Altrimenti procedo con l'esplorazione passando per 
        # gli iperonimi del senso passato come parametro
        sense_min_depth = 1 + min(min_depth_through_node(x, node) for x in hypernyms)
    
    return sense_min_depth



@lru_cache(maxsize=256)
def get_all_hypernyms(sense):
    # Ottengo tutti gli iperonimi (fino alla radice) di sense
    # Creo un insieme contenente tutti gli iperonimi del senso passato in input
    hypernyms = set(sense.hypernyms() + sense.instance_hypernyms() + [sense])
    tmp = set()

    # Se ho un solo elemento nell'insieme vuol dire che ho raggiunto la radice
    if len(hypernyms) == 1:
        return hypernyms
    else:
        # Altrimenti procedo ricorsivamente col recupero degli iperonimi degli iperonimi
        # e li aggiungo all'insieme di iperonimi di sense
        for hypernym in hypernyms:
            if hypernym != sense:
                tmp = tmp.union(get_all_hypernyms(hypernym))
        hypernyms = hypernyms.union(tmp)
    
    return hypernyms


@lru_cache(maxsize=256)
def find_lcs(sense_a, sense_b):
    # Ottengo l'intersezione di tutti gli iperonimi in comune per entrambi i sensi
    common_hypernyms = get_all_hypernyms(sense_a).intersection(get_all_hypernyms(sense_b))
    if len(common_hypernyms) != 0:
        # Ottengo la profondità dell'iperonimo comune che ha la minima profondità massimizzata
        lcs_depth = max(min_depth(x) for x in common_hypernyms)
        # Ritorno una lista ordinata di nodi
        return sorted([x for x in common_hypernyms if lcs_depth == min_depth(x)])
    return []

@lru_cache(maxsize=256)
def hypernym_distances(sense):
    # Ritorna un dizionario con chiave il senso (iperonimo di sense), e valore la
    # altezza da sense alla prima occorrenza dell'iperonimo
    # Creo una coda contenente il senso passato in input e la profondita'
    queue = deque([(sense, 0)])
    path = {}
    
    while len(queue) != 0:
        # Estraggo il primo elemento sinistro della coda
        cur_sense, depth = queue.popleft()
        
        if cur_sense in path:
            continue
        
        # Imposto la altezza del senso (iperonimo di sense) uguale
        # al livello attuale (visita in ampiezza)
        path[cur_sense] = depth
        depth += 1
        
        # Espando la coda inserendo gli iperonimi del senso attuale 
        # e ricomincio il ciclo aumentando la profondità di volta
        # in volta
        queue.extend((hypernym, depth) for hypernym in cur_sense.hypernyms())
        queue.extend((hypernym, depth) for hypernym in cur_sense.instance_hypernyms())
        
    return path

@lru_cache(maxsize=256)
def shortest_path_distance(sense1, sense2):
       
    if sense1 == sense2:
        minpath = 0
    else:
        minpath = float('inf')
        hyp_list1 = hypernym_distances(sense1)
        hyp_list2 = hypernym_distances(sense2)
        # Calcolo il percorso minimo sommando le distanze dai sensi ai
        # loro iperonimi comuni e ottenendo la minima somma
        for key1, value1 in hyp_list1.items():
            value2 = hyp_list2.get(key1, float("inf"))
            minpath = min(minpath, value1 + value2)

    return minpath
