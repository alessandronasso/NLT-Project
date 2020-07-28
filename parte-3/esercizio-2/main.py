from utils import *
from tqdm import tqdm

DATASET = 'content-to-form.xlsx'
MAX_DEPTH = 8
MAX_GENUS_CANDIDATES = 20

if __name__ == '__main__':

    dataset = load_and_preprocess(DATASET)

    definition_to_sense = {}

    max_concept_size = 0
    max_synset_name_size = 0

    for concetto, row in tqdm(dataset.iterrows(), total=len(dataset), unit='concept'):
        sense, _ = wsi(dataset.loc[concetto, 'sense_counts'], dataset.loc[concetto, 'word_freq'], MAX_DEPTH, MAX_GENUS_CANDIDATES)
        definition_to_sense[concetto] = sense
        max_concept_size = max(max_concept_size, len(concetto))
        max_synset_name_size = max(max_synset_name_size, len(sense.name()))
    
    print()
    for concetto, senso in definition_to_sense.items():
        print(f'{concetto:<{max_concept_size}} Synset: {senso.name():<{max_synset_name_size}} Definition: {senso.definition()}')
    print()
