import pandas as pd
from nltk.corpus import wordnet as wn
from common import WORDSIM_PATH, find_max_similarity
from similarity import wup_similarity, path_similarity, lch_similarity
from scipy.stats import spearmanr, pearsonr

if __name__ == '__main__':
    word_sim_df = pd.read_csv(WORDSIM_PATH, index_col=['Word 1', 'Word 2']).sort_index()

    for index, row in word_sim_df.iterrows():
        synsets1 = wn.synsets(index[0])
        synsets2 = wn.synsets(index[1])
        word_sim_df.loc[index, 'wup_similarity'] = find_max_similarity(synsets1, synsets2, wup_similarity)
        word_sim_df.loc[index, 'path_similarity'] = find_max_similarity(synsets1, synsets2, path_similarity)
        word_sim_df.loc[index, 'lch_similarity'] = find_max_similarity(synsets1, synsets2, lch_similarity)

    print(word_sim_df.to_string())
    print("\n\n")
    spearman_wup = spearmanr(word_sim_df.loc[:, 'Human (mean)'].values, word_sim_df.loc[:, 'wup_similarity'].values)[0]
    pearson_wup = pearsonr(word_sim_df.loc[:, 'Human (mean)'].values, word_sim_df.loc[:, 'wup_similarity'].values)[0]
    spearman_path = spearmanr(word_sim_df.loc[:, 'Human (mean)'].values, word_sim_df.loc[:, 'path_similarity'].values)[0]
    pearson_path = pearsonr(word_sim_df.loc[:, 'Human (mean)'].values, word_sim_df.loc[:, 'path_similarity'].values)[0]
    spearman_lch = spearmanr(word_sim_df.loc[:, 'Human (mean)'].values, word_sim_df.loc[:, 'lch_similarity'].values)[0]
    pearson_lch = pearsonr(word_sim_df.loc[:, 'Human (mean)'].values, word_sim_df.loc[:, 'lch_similarity'].values)[0]
    
    print(f'Wu & Palmer Similarity Pearson Correlation Coefficient: {pearson_wup}')
    print(f'Wu & Palmer Similarity Spearman Correlation Coefficient: {spearman_wup}')
    print(f'Path Similarity Pearson Correlation Coefficient: {pearson_path}')
    print(f'Path Similarity Spearman Correlation Coefficient: {spearman_path}')
    print(f'Leakcock & Chodorow Similarity Pearson Correlation Coefficient: {pearson_lch}')
    print(f'Leakcock & Chodorow Similarity Spearman Correlation Coefficient: {spearman_lch}')
    print('\n')