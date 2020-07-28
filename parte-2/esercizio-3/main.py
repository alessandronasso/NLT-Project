from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
from utils import read_annotations, load_stopwords
from disambiguation import lesk_disambiguate
import pandas as pd

# FRAMES
#
# Alternatives
# Controller_object (Controller)
# Want_suspect (Want)
# Building_subparts (Building)
# Hedging
# Be_in_agreement_on_action (agreement)
# Disgraceful_situation (situation)
# Change_event_duration (event)
# Intentional_deception (deception)

ANNOTATIONS_PATH = 'annotations.csv'
STOPWORDS_PATH = 'stop_words_FULL.txt'

if __name__ == '__main__':
    annotations = read_annotations(ANNOTATIONS_PATH)
    stopw = load_stopwords(STOPWORDS_PATH)

    res_rows = []

    giuste = 0
    for frame, word, target_synset in annotations:
        # Se non è disponibile il mapping
        if target_synset is None:
            giuste += 1
            continue

        # Rimuovo il PoS dalla parola estratta dalle annotazioni
        input_word = word.split('.')[0]
        # disambiguo la parola input_word (quella centrale
        # nelle annotazioni) usando il frame come contesto
        synset = lesk_disambiguate(frame, input_word, stopwordset=stopw)

        if synset == target_synset:
            giuste += 1
        
        res_rows.append({
            'Frame': frame.name, 
            'Word': word, 
            'Disambiguated Sense': synset.name(), 
            'Target Sense': target_synset.name(),
            'Correct Answer': synset == target_synset
        })
    
    print()
    print(pd.DataFrame(res_rows).to_string())

    accuracy = giuste / len(annotations) * 100

    print()
    print(f'Accuracy: {accuracy:.2f} %')
    print()

