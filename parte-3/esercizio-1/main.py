import pandas as pd
from utils import preprocess, calc_mean_similarity, similarity

DATASET = 'definizioni.xlsx'

def main():
    # Carico dataset
    dataset = pd.read_excel(DATASET)
    dataset.columns = ['id', 'concreto_generico', 'concreto_specifico', 'astratto_generico', 'astratto_specifico']
    dataset = dataset.set_index('id')

    # Preprocessing frasi
    dataset['concreto_generico'] = dataset['concreto_generico'].map(preprocess, na_action='ignore')
    dataset['concreto_specifico'] = dataset['concreto_specifico'].map(preprocess, na_action='ignore')
    dataset['astratto_generico'] = dataset['astratto_generico'].map(preprocess, na_action='ignore')
    dataset['astratto_specifico'] = dataset['astratto_specifico'].map(preprocess, na_action='ignore')

    results = pd.DataFrame(columns=['astratto', 'concreto'])

    # Calcolo similarità media tra le definizioni dei 4 concetti
    res_concreto_generico = calc_mean_similarity(dataset[dataset['concreto_generico'].notnull()]['concreto_generico'], similarity)
    res_concreto_specifico = calc_mean_similarity(dataset[dataset['concreto_specifico'].notnull()]['concreto_specifico'], similarity)
    res_astratto_generico = calc_mean_similarity(dataset[dataset['astratto_generico'].notnull()]['astratto_generico'], similarity)
    res_astratto_specifico = calc_mean_similarity(dataset[dataset['astratto_specifico'].notnull()]['astratto_specifico'], similarity)

    results.loc['generico'] = [res_astratto_generico, res_concreto_generico]
    results.loc['specifico'] = [res_astratto_specifico, res_concreto_specifico]

    print()
    print(results)
    print()

if __name__ == '__main__':
    main()