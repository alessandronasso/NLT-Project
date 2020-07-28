from lesk import lesk_disambiguate
from utils import read_sentence_txt, SENTENCES_PATH, load_stopwords

STOPWORDS_PATH = 'stop_words_FULL.txt'

if __name__ == '__main__':
    sentences = read_sentence_txt(SENTENCES_PATH)
    stopw = load_stopwords(STOPWORDS_PATH)

    for sentence, index, original in sentences:
        # Passo come parametro la frase e l'indice con l'elemento
        # da sostituire
        sense = lesk_disambiguate(sentence, index, stopwordset=stopw)
        print('Sentence:', original)
        print('Word:', sentence[index], 'at index ', index)
        print('Synset Name:', sense.name())
        print('Synset Definition:', sense.definition())
        print('Synonyms:', sense.lemma_names())
        new_sent = sentence
        new_sent[index] = str(sense.lemma_names())
        print('Rewritten Sentence:', ' '.join(new_sent))
        print('\n\n')