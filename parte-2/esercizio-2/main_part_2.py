from lesk import lesk_disambiguate
from utils import extract_semcor_sentences
from nltk.corpus import semcor
from utils import load_stopwords
import random

# seme per il generatore di numeri casuali, usato per scegliere i sostantivi
SEED = 117
COUNT = 50
START = 0
STOPWORDS_PATH = 'stop_words_FULL.txt'

if __name__ == '__main__':
    random.seed(SEED)

    giuste = 0
    semcor_sentences = extract_semcor_sentences(count=COUNT, start=START)
    stopw = load_stopwords(STOPWORDS_PATH)

    for i, tagged_sentence, nn_words in semcor_sentences:
        # Scelgo casualmente un sostantivo dalla frase
        # lemma contiene l'oggetto di tipo Lemma che contiene il nome del synset
        # word contiene la parola in sè
        lemma, word = random.choice(nn_words)
        sentence = semcor.sents()[i]
        # Effettuo la disambiguazione passandogli la frase relativa
        # e la parola da disambiguare, ottenendo il senso migliore
        syn = lesk_disambiguate(sentence, word, stopwordset=stopw)
        # Se il synset ottenuto con l'algoritmo di lesk è uguale al
        # synset preso dall'annotazione in semcor, incremento il
        # numero di frasi corrette
        if lemma.synset() == syn:
            giuste += 1
    
    accuracy = giuste / COUNT * 100
    
    print(f'Accuracy: {accuracy:.2f} %')