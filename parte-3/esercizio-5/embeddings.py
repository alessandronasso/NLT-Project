from gensim.models import KeyedVectors
import pickle


try:
    vector_file_pickle = open('numberbatch-en.pkl', 'rb')
    print("Loading word vectors from pickle dump file...")
    word_vectors = pickle.load(vector_file_pickle)
    vector_file_pickle.close()
except:
    print("Loading word vectors...")
    word_vectors = KeyedVectors.load_word2vec_format('numberbatch-en.txt', binary=False)
    vector_file_pickle = open('numberbatch-en.pkl', 'wb')
    print("Saving word vectors to pickle file...")
    pickle.dump(word_vectors, vector_file_pickle)
    vector_file_pickle.close()
print("Done.")