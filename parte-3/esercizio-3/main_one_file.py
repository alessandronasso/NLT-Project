from utils import *
from tqdm import tqdm
from collections import Counter, defaultdict

# DATASET URL
# http://www.statmt.org/europarl/

VERB = 'buy'

SENTENCES_FILE_PATHS = 'europarl/buy.1504.txt'
MAX_SENTENCES = 50000

if __name__ == '__main__':
    semantic_clusters = Counter()

    sentences = read_sentences(SENTENCES_FILE_PATHS)
    sentence_count = len(sentences)
    sentence_count = min(sentence_count, MAX_SENTENCES)
    sentences = sentences[:sentence_count]

    pbar = tqdm(total=sentence_count, unit=' sentences', leave=False)

    for doc in nlp.pipe(sentences, n_threads=8, batch_size=2500):
        for token in doc:
            if token.lemma_ == VERB and token.pos_ == 'VERB':
                args = get_verb_args(doc, token)
                if len(args) > 0:
                    try:
                        subj, obj = args
                        subj_supersense = lesk_disambiguate(subj).lexname()
                        obj_supersense = lesk_disambiguate(obj).lexname()
                        semantic_clusters[(subj_supersense, obj_supersense)] += 1
                    except IndexError as e:
                        pass
        pbar.update()
    
    total_count = sum(semantic_clusters.values())
    max_val_digits = len(str(max(semantic_clusters.values())))

    subjects = defaultdict(int)
    objects = defaultdict(int)

    print('\n')
    for cluster, count in sorted(semantic_clusters.items(), key=lambda x:x[1], reverse=True):
        supersense_subj = cluster[0].split('.')[1]
        supersense_obj = cluster[1].split('.')[1]
        subjects[supersense_subj] += count
        objects[supersense_obj] += count
        print(f'Subject: {supersense_subj:<13} Object: {supersense_obj:<13} Count: {count:<{max_val_digits}} Frequency: {count / total_count * 100:>5.2f} %')
    
    print('\nList of all subjects:\n')
    for subj, count in sorted(subjects.items(), key=lambda x:x[1], reverse=True):
        print(f'Subject: {subj:<13} Count: {count:<{len(str(max(subjects.values())))}} Frequency: {count / sum(subjects.values()) * 100:>5.2f} %')
    
    print('\nList of all objects:\n')
    for obj, count in sorted(objects.items(), key=lambda x:x[1], reverse=True):
        print(f'Subject: {obj:<13} Count: {count:<{len(str(max(objects.values())))}} Frequency: {count / sum(objects.values()) * 100:>5.2f} %')
   
    print(f'\nNumero totale di frasi esaminate: {total_count}')
    print(f'Numero totale di cluster semantici: {len(semantic_clusters)}\n')