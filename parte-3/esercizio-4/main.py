from utils import *
import pandas as pd
import matplotlib.pyplot as plt

ARTICLE_PATH = 'article2.txt'
STOPWORDS_PATH = 'stop_words_FULL.txt'
WINDOW_SIZE_MULTIPLIER = 1            # Se 1 la finestra è grande quanto la dimensione media di un paragrafo
SIMILARITY_CLUSTERING_THRESHOLD = 1.1 # Se > 1 clustering disabilitato
SIMILARITY_FUNCTION_FOR_CLUSTERING = lambda x, y: x.wup_similarity(y)
SIDE_EXCLUSION_ZONE_SIZE = 2
PLOT_SIMILARITY_CHART = True


if __name__ == '__main__':
    sentences, paragraphs = read_article(ARTICLE_PATH)

    window_size = round(len(sentences)/len(paragraphs) * WINDOW_SIZE_MULTIPLIER)

    stopw = load_stopwords(STOPWORDS_PATH)

    sentence_dict, all_senses = preprocess_sentences(sentences, stopw)

    cluster_indexes = cluster_senses(all_senses, SIMILARITY_FUNCTION_FOR_CLUSTERING, SIMILARITY_CLUSTERING_THRESHOLD)

    topic_matrix = create_topic_matrix(sentence_dict, cluster_indexes)

    similarity_data = rolling_window_similarity(topic_matrix, window_size)

    split_indexes, minimas = choose_split_indexes(similarity_data, window_size, SIDE_EXCLUSION_ZONE_SIZE)

    if PLOT_SIMILARITY_CHART:
        pd.Series(similarity_data).plot()
        for xc in minimas:
            plt.axvline(x=xc, color='red')
        plt.show()

    print()
    for i, sent in enumerate(sentences):
        print(sent)
        if i+1 in split_indexes:
            print('---------------------------------------------------------------------------')
    print()
