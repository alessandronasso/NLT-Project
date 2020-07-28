from utils import *
from summarization import *
from settings import *
import time, os

STOPWORDS = 'stop_words_FULL.txt'
DOCUMENTS_FOLDER = 'documents'

if __name__ == '__main__':
    stopw = load_stopwords(STOPWORDS)
    opzioni_nasari = {'big': NASARI_L_SETTINGS, 'small': NASARI_S_SETTINGS}

    scelta_nasari = ask('Scegliere il file NASARI desiderato (big/small): ', opzioni_nasari.keys())

    SETTINGS = opzioni_nasari[scelta_nasari]
    with timing(message='Vettori NASARI caricati e preprocessati in'):
        nasari = (load_nasari(**SETTINGS), SETTINGS['dim'])

    files = tuple(os.listdir(DOCUMENTS_FOLDER))
    choose_file_question = '\n'
    for i, f in enumerate(files):
        choose_file_question += f'[{i}] {f}\n'
    choose_file_question += '\nScegli un file da riassumere (back per uscire): '

    while True:
        file_scelto = ask(choose_file_question, ['back'] + [str(i) for i in range(len(files))])
        if file_scelto == 'back':
            break
        
        file_scelto = files[int(file_scelto)]
        file_scelto = os.path.join(DOCUMENTS_FOLDER, file_scelto)

        personalize = ask('Vuoi personalizzare le opzioni (s/n): ', ['s', 'n'])
        
        options = {
            'stopw': stopw,
            'sig_size': 20,
            'title_weight': 5,
            'first_sentence_bonus': 1.3,
            'last_sentence_bonus': 1.05
        }

        if personalize == 's':
            sig_size = ask('Dimensione signature del topic (numero intero, defalut=20): ', validator=lambda x: x.isdigit())
            title_weight = ask('Peso del titolo sulla signature (numero reale, default=5): ', validator=isfloat)
            first_sentence_bonus = ask('Bonus prima frase nel paragrafo (numero reale, default=1.3): ', validator=isfloat)
            last_sentence_bonus = ask('Bonus ultima frase nel paragrafo (numero reale, default=1.05): ', validator=isfloat)
        
            options = {
                'stopw': stopw,
                'sig_size': int(sig_size),
                'title_weight': float(title_weight),
                'first_sentence_bonus': float(first_sentence_bonus),
                'last_sentence_bonus': float(last_sentence_bonus)
            }

        doc = load_document(file_scelto, nasari, options_dict=options)

        change_compression = True

        while change_compression:
            compression = ask('Scegli un livello di compressione (reale tra 0 e 1, back per tornare alla scelta del file): ', validator=lambda x: x == 'back' or (isfloat(x) and 0 <= float(x) and float(x) <= 1))
            if compression == 'back':
                change_compression = False
                continue
            compression = float(compression)
            print("\n-------------------------------------- INIZIO RIASSUNTO --------------------------------------\n")
            with timing(message='Riassunto generato in'):
                print(doc.summary(compression_ratio=compression))
            print("\n--------------------------------------- FINE RIASSUNTO ---------------------------------------\n")
        