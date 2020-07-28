from common import *
import math

def wup_similarity(sense_a, sense_b):
    similarity = 0
    # Cerco la lista di nodi in comune con profondità
    # minima tra i due sensi 
    lcs_list = find_lcs(sense_a, sense_b)
    if len(lcs_list) != 0:
        # Se il senso corrente è nella lista estraggo quello,
        # altrimenti prendo il primo (essendo ordinati)
        LCS = sense_a if sense_a in lcs_list else lcs_list[0]
        # Ottengo la distanza dalla root del nodo in comune
        depth = max_depth(LCS) + 1
        # Ottengo la profondità dal sense_a alla root passando per l'LCS
        depth_sense_a = min_depth_through_node(sense_a, LCS) + depth
        # Ottengo la profondità dal sense_b alla root passando per l'LCS
        depth_sense_b = min_depth_through_node(sense_b, LCS) + depth
        # Eseguo il calcolo della similarità
        similarity = (2.0 * depth) / (depth_sense_a + depth_sense_b) 
    return similarity


def path_similarity(sense_a, sense_b):
    # Cerco la distanza minima tra i due sensi 
    # ed applico la formula
    minpath = shortest_path_distance(sense_a, sense_b)
    if minpath == float("inf"):
        minpath = 2 * MAX_DEPTH - 1
    return 2 * MAX_DEPTH - (1 + minpath)


def lch_similarity(sense_a, sense_b):
    # Cerco la distanza minima tra i due sensi 
    # ed applico la formula
    minpath = shortest_path_distance(sense_a, sense_b)
    if minpath == float("inf"):
        minpath = 2 * MAX_DEPTH
    return - math.log((minpath + 1) / (2.0 * MAX_DEPTH + 1))