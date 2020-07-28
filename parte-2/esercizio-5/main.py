from utils import *
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import csv

DATI_ANNOTATI_PARTE_1 = 'coppie.tsv'
DATI_ANNOTATI_PARTE_2_L = 'coppie-luca-synsets.tsv'
DATI_ANNOTATI_PARTE_2_A = 'coppie-ale-synsets.tsv'
NASARI_PATH = 'mini_NASARI.tsv'
WORDS_SYNSETS_FILE_PATH = 'SemEval17_IT_senses2synsets.txt'

if __name__ == '__main__':
    # Caricamento dei dati annotati a mano
    dati_annotati_1 = pd.read_csv(DATI_ANNOTATI_PARTE_1, delimiter='\t', index_col=['Parola_1', 'Parola_2'])
    dati_annotati_2_l = pd.read_csv(DATI_ANNOTATI_PARTE_2_L, delimiter='\t', quoting=csv.QUOTE_NONE, index_col=['Parola_1', 'Parola_2'])
    dati_annotati_2_a = pd.read_csv(DATI_ANNOTATI_PARTE_2_A, delimiter='\t', quoting=csv.QUOTE_NONE, index_col=['Parola_1', 'Parola_2'])
    nasari = load_nasari(NASARI_PATH)
    # Dizionario formato da parola + dizionario con BS/vec
    word2synsets = load_synset_ids(WORDS_SYNSETS_FILE_PATH, nasari)
    
    # Calcolo degli agreement tra le annotazioni dei 2 annotatori
    inter_rater_agreement_pearson = dati_annotati_1[['Similarita_1', 'Similarita_2']].corr(method='pearson').loc['Similarita_1', 'Similarita_2']
    inter_rater_agreement_spearman = dati_annotati_1[['Similarita_1', 'Similarita_2']].corr(method='spearman').loc['Similarita_1', 'Similarita_2']

    print('\nCONSEGNA 1\n')
    print(f'Inter-rater agreement (Pearson): {inter_rater_agreement_pearson:.2f}')
    print(f'Inter-rater agreement (Spearman): {inter_rater_agreement_spearman:.2f}')

    # Estrazione dei synset della parola della prima colonna
    synsets_parola_1 = lambda x: list(word2synsets[x[0]].values())
    # Estrazione dei synset della parola della seconda colonna
    synsets_parola_2 = lambda x: list(word2synsets[x[1]].values())
    # Calcolo della massima cosine similarity su tutti i sensi estratti
    max_similarity_func = lambda x: max_cosine_similarity(synsets_parola_1(x), synsets_parola_2(x))

    # Aggiungo al dataframe la cosine similarity calcolata tra le coppie di parole
    dati_annotati_1['Cosine_similarity'] = dati_annotati_1.index.map(max_similarity_func)
    
    # Eseguo Pearson e Spearman sulla similarità media e la cosine similarity precedentemente calcolata
    valutazione_pearson = dati_annotati_1[['Similarita_media', 'Cosine_similarity']].corr(method='pearson').loc['Similarita_media', 'Cosine_similarity']
    valutazione_spearman = dati_annotati_1[['Similarita_media', 'Cosine_similarity']].corr(method='spearman').loc['Similarita_media', 'Cosine_similarity']

    print(f'Valutazione (Pearson): {valutazione_pearson:.2f}')
    print(f'Valutazione (Spearman): {valutazione_spearman:.2f}')
    print()

    print('CONSEGNA 2\n')

    # Estraggo tutte le label per ogni colonna
    labels_parola_1, labels_parola_2 = get_all_labels(dati_annotati_1.index, word2synsets)

    # Calcolo k-cohen sui babelsynset annotati a mano per ogni parola
    k_cohen_parola_1 = cohen_kappa_score(dati_annotati_2_l['BS1'].values, dati_annotati_2_a['BS1'].values, labels=tuple(labels_parola_1))
    k_cohen_parola_2 = cohen_kappa_score(dati_annotati_2_l['BS2'].values, dati_annotati_2_a['BS2'].values, labels=tuple(labels_parola_2))

    print(f'Inter-rater agreement prima parola (Kappa di Cohen): {k_cohen_parola_1:.2f}')
    print(f'Inter-rater agreement seconda parola (Kappa di Cohen): {k_cohen_parola_2:.2f}')
    print(f'Inter-rater agreement medio (Kappa di Cohen): {(k_cohen_parola_1 + k_cohen_parola_2) / 2:.2f}')

    # Ottengo la coppia di synset che massimizza la cosine similarity (disambiguazione)
    output_algo = dati_annotati_1['Cosine_similarity'].to_frame()
    most_similar_synsets_func = lambda x: find_most_similar_synsets(synsets_parola_1(x), synsets_parola_2(x), cosine_similarity, dati_annotati_1.loc[x, 'Cosine_similarity'])

    for index, row in output_algo.iterrows():
        res = most_similar_synsets_func(index)
        output_algo.loc[index, 'BS1'] = res[0]
        output_algo.loc[index, 'BS2'] = res[1]

    # Calcolo la accuracy della disambiguazione con i dati annotati dal primo annotatore (Luca)
    acc_parola_1_l = dati_annotati_2_l[dati_annotati_2_l['BS1'] == output_algo['BS1']]['BS1'].count() / output_algo['BS1'].count()
    acc_parola_2_l = dati_annotati_2_l[dati_annotati_2_l['BS2'] == output_algo['BS2']]['BS2'].count() / output_algo['BS2'].count()
    acc_both_l = dati_annotati_2_l[(dati_annotati_2_l['BS1'] == output_algo['BS1']) & (dati_annotati_2_l['BS2'] == output_algo['BS2'])]['BS1'].count() / output_algo['BS1'].count()

    # Calcolo la accuracy della disambiguazione con i dati annotati dal secondo annotatore (Alessandro)
    acc_parola_1_a = dati_annotati_2_a[dati_annotati_2_a['BS1'] == output_algo['BS1']]['BS1'].count() / output_algo['BS1'].count()
    acc_parola_2_a = dati_annotati_2_a[dati_annotati_2_a['BS2'] == output_algo['BS2']]['BS2'].count() / output_algo['BS2'].count()
    acc_both_a = dati_annotati_2_a[(dati_annotati_2_a['BS1'] == output_algo['BS1']) & (dati_annotati_2_a['BS2'] == output_algo['BS2'])]['BS1'].count() / output_algo['BS1'].count()

    print(f'Accuratezza parola 1 (Luca M.): {acc_parola_1_l * 100:.2f} %')
    print(f'Accuratezza parola 2 (Luca M.): {acc_parola_2_l * 100:.2f} %')
    print(f'Accuratezza coppia parole (Luca M.): {acc_both_l * 100:.2f} %')

    print(f'Accuratezza parola 1 (Alessandro N.): {acc_parola_1_a * 100:.2f} %')
    print(f'Accuratezza parola 2 (Alessandro N.): {acc_parola_2_a * 100:.2f} %')
    print(f'Accuratezza coppia parole (Alessandro N.): {acc_both_a * 100:.2f} %')
    print()
