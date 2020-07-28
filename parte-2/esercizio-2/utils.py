from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma
from nltk.tokenize import word_tokenize
from nltk.corpus import semcor
from itertools import islice
import string

SENTENCES_PATH = 'sentences.txt'

def get_nn_words(sentence):
    # Esploro l'albero della frase per cercare
    # i sostantivi nella frase ed estrarre tutti
    # i loro synsets e il lemma
    nn_words = []
    for tree in sentence:
        if type(tree.label()) == Lemma:
            child = tree[0]
            if child.label() == 'NN' and len(child) == 1:
                if len(wn.synsets(child[0])) > 0:
                    nn_words.append((tree.label(), child[0]))
                else:
                    new_word = ''.join([x for x in child[0] if not x in string.punctuation])
                    if len(wn.synsets(new_word)) > 0:
                        nn_words.append((tree.label(), new_word))
    return nn_words
    
def window(seq, n=2):
    # Ritorna una finestra mobile di dimensione n sulla sequenza seq passata.
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
def rejoin_asteriscs_words(sentence):
    new_sentence = []
    next_index = 0
    size = len(sentence)
    for i, w in enumerate(window(sentence, n=5)):
        if w[0] == '*' and w[1] == '*' and w[3] == '*' and w[4] == '*':
            new_sentence.append(f'**{w[2]}**')
            next_index += 5
            size -= 4
        elif next_index == i:
            new_sentence.append(sentence[i])
            next_index += 1
    if len(new_sentence) < size:
        difference = size - len(new_sentence)
        new_sentence = new_sentence + sentence[-difference:]
    return new_sentence

def read_sentence_txt(path):
    # Metodo che permette di caricare il file contenente
    # le frasi
    with open(path, 'r', encoding="utf-8") as f:
        sentences = [line[:-1][2:].strip() for line in f.readlines() if line.startswith('- ')]
    
    original_sentences = sentences
    sentences = [rejoin_asteriscs_words(word_tokenize(x)) for x in sentences]

    # Creo una copia della frase rimuovendo ** e salvando
    # l'indice della parola da sostituire
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if word.startswith('**') and word.endswith('**'):
                word = word.replace('**', '')
                sentence[j] = word
                sentences[i] = (sentence, j, original_sentences[i])

    return sentences

def extract_semcor_sentences(count=50, start=0):
    semcor_sentences = []
    # Estraggo la frase dal SemCor sottoforma di chunks, ognuno dei
    # quali con il proprio PoS
    for i, sentence in enumerate(semcor.tagged_sents(tag='both')[start:]):
        #Cerco i sostantivi nella frase
        nn_words = get_nn_words(sentence)
        if len(nn_words) > 0:
            semcor_sentences.append((i + start, sentence, nn_words))
        if len(semcor_sentences) == count:
            break
    return semcor_sentences
    
def load_stopwords(path):
    # Metodo che permette di caricare il file contenente
    # le stopwords
    if path == None: return None
    with open(path, 'r', encoding="utf-8") as f:
        words = [line[:-1] for line in f.readlines()]
    return set(words)