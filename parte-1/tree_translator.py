from it_en_dict import it_en_dict
from simplenlg import *

def parse_s(tree, nlgFactory):
    left_child = tree[0]
    s = nlgFactory.createClause()

    if left_child.label() == 'VP':
        # S -> VP NP
        vp, subj, tense, perfect, passive = parse_vp(tree[0], nlgFactory)
        np = parse_np(tree[1], nlgFactory)
        s.setSubject(subj)
        s.setVerb(vp)
        s.setObject(np)
    elif left_child.label() == 'NP':
        # S -> NP VP
        np = parse_np(tree[0], nlgFactory)
        vp, subj, tense, perfect, passive = parse_vp(tree[1], nlgFactory)
        if passive:
            s.setObject(np)
        else:
            s.setSubject(np)
        s.setVerb(vp)
    
    s.setFeature(Feature.TENSE, tense)
    s.setFeature(Feature.PERFECT, perfect)
    s.setFeature(Feature.PASSIVE, passive)
    return s

def parse_vp(tree, nlgFactory):
    vp, subj, tense, perfect, passive = parse_verb(tree[0], nlgFactory) # VP -> Verb
    if len(tree) == 2:
        right_child = tree[1]
        if right_child.label() == 'NP':
            # VP -> Verb NP
            np = parse_np(tree[1], nlgFactory)
            vp.setObject(np)
        elif right_child.label() == 'Adv':
            # VP -> Verb Adv
            adv, pos = it_en_dict[tree[1][0]]
            vp.addPostModifier(adv)
    elif len(tree) == 3:
        # VP -> Verb Adv NP
        adv, pos = it_en_dict[tree[1][0]]
        vp.addPostModifier(adv)
        np = parse_np(tree[2], nlgFactory)
        vp.setObject(np)
    return vp, subj, tense, perfect, passive

def parse_np(tree, nlgFactory):
    left_child = tree[0]
    if left_child.label() == 'Art':
        # NP -> Art Nominal
        art, pos = it_en_dict[tree[0][0]]
        np = parse_nominal(tree[1], nlgFactory)
        np.setDeterminer(art)
    elif left_child.label() == 'AP':
        # NP -> AP Nominal
        ap, pos = it_en_dict[tree[0][0]]
        np = parse_nominal(tree[1], nlgFactory)
        np.addPreModifier(ap)
    return np

def parse_verb(tree, nlgFactory):
    if len(tree) == 1:
        # Verb -> V
        v = tree[0]
        verb, pos, subj, tense, perfect, passive = it_en_dict[v[0]]
        vp = nlgFactory.createVerbPhrase(verb)
    elif len(tree) == 2:
        # Verb -> Aux V
        aux = tree[0]
        v = tree[1]
        verb, pos, subj, tense, perfect, passive = it_en_dict[aux[0] + '-' + v[0]]
        vp = nlgFactory.createVerbPhrase(verb)
    elif len(tree) == 3:
        # Verb -> Aux Aux V
        aux1 = tree[0]
        aux2 = tree[1]
        v = tree[2]
        verb, pos, subj, tense, perfect, passive = it_en_dict[aux1[0] + '-' + aux2[0] + '-' + v[0]]
        vp = nlgFactory.createVerbPhrase(verb)
    return vp, subj, tense, perfect, passive

def parse_nominal(tree, nlgFactory):
    if len(tree) == 1:
        # Nominal -> Noun
        noun, pos, plural = it_en_dict[tree[0][0]]
        np = nlgFactory.createNounPhrase(noun)
        np.setFeature(Feature.NUMBER, plural)
        return np
    elif len(tree) == 2:
        left_child, right_child = tree[0], tree[1]
        if left_child.label() == 'Nominal':
            np = parse_nominal(tree[0], nlgFactory)
            if right_child.label() == 'Adj':
                # Nominal -> Nominal Adj
                adj, pos = it_en_dict[tree[1][0]]
                np.setPreModifier(adj)
            elif right_child.label() == 'PP':
                # Nominal -> Nominal PP
                pp = parse_pp(tree[1], nlgFactory)
                np.addComplement(pp)
        elif left_child.label() == 'Noun':
            # Nominal -> Noun Nominal
            noun, pos, plural = it_en_dict[tree[0][0]]
            np = parse_nominal(tree[1], nlgFactory)

            parola_composta = noun + '-' + np_to_string(np)
            if parola_composta in it_en_dict.keys():
                n, pos, pl = it_en_dict[parola_composta]
                np.setFeature(Feature.NUMBER, pl)
                np.setNoun(n)
            else:
                np.setPreModifier(noun)
        elif left_child.label() == 'Adj':
            # Nominal -> Adj Nominal
            adj, pos = it_en_dict[tree[0][0]]
            np = parse_nominal(tree[1], nlgFactory)
            np.setPreModifier(adj)
        return np

def parse_pp(tree, nlgFactory):
    right_child = tree[1]
    preposition, pos = it_en_dict[tree[0][0]]
    
    if right_child.label() == 'NP':
        # PP -> Preposition NP
        np = parse_np(tree[1], nlgFactory)
    elif right_child.label() == 'Nominal':
        # PP -> Preposition Nominal
        np = parse_nominal(tree[1], nlgFactory)
    
    pp = nlgFactory.createPrepositionPhrase()
    pp.setPreposition(preposition)
    pp.addComplement(np)
    return pp

def np_to_string(np):
    s = str(np)
    return s.split('[')[1].split(':')[0]
