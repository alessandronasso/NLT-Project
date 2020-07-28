from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

lemmatizer = WordNetLemmatizer()


def ctx(sentence, stopwordset=None):
    # Splitto le singole parole della frase
    sentence = word_tokenize(sentence)
    # Specifico gli elementi problematici che dovrò
    # escludere in seguito
    blacklist = ["''", "``"] + list(string.punctuation)
    
    if stopwordset == None:
        stopwordset = set(stopwords.words('english'))
    
    # Lemmatizzo tutte le parole che non sono nella blacklist e che non
    # hanno una forma numerica o particolare
    context = {lemmatizer.lemmatize(x.lower()) for x in sentence if not x.lower() in stopwordset and not x in blacklist and not x.isnumeric() and x != ''}
    context = {x[1:] if x.startswith('\'') else x for x in context}
    context = {x for x in context if not x in stopwordset and x != ''}
    return context


def get_signature(sense, stopwordset=None, expand=True):
    # Per ottenere la signature splitto la frase in singole 
    # parole tramite word_tokenize e poi estraggo il contesto
    signature = ctx(sense.definition(), stopwordset)
    
    # Effettuo la stessa operazione per gli esempi del sysnet 
    # presenti in wordnet
    for example in sense.examples():
        signature = signature.union(ctx(example, stopwordset))
    
    # Aggiungo i sinonimi lemmatizzati   
    synonyms = {lemmatizer.lemmatize(x.lower()) for x in sense.lemma_names() if not '_' in x}
    signature = signature.union(synonyms)

    # Per migliorare la precisione dell'algoritmo aggiungo
    # inoltre gli iperonimi, gli iponimi e i meronimi alla
    # signature
    if expand:
        hypernym_sig, hyponym_sig, meronym_sig = set(), set(), set()
        
        for hypernym in sense.hypernyms() + sense.instance_hypernyms():
            hypernym_sig = hypernym_sig.union(get_signature(hypernym, stopwordset, False))
            
        for hyponym in sense.hyponyms() + sense.instance_hyponyms():
            hyponym_sig = hyponym_sig.union(get_signature(hyponym, stopwordset, False))
            
        for meronym in sense.part_meronyms() + sense.member_meronyms():
            meronym_sig = meronym_sig.union(get_signature(meronym, stopwordset, False))
        
        signature = signature.union(hypernym_sig)
        signature = signature.union(hyponym_sig)
        signature = signature.union(meronym_sig)
    
    return signature


def frame_ctx(frame, stopwordset=None):
    context = ctx(frame.definition, stopwordset)
    
    for name, fe in frame.FE.items():
        context = context.union(ctx(fe.definition, stopwordset))
    
    return context


def lesk_disambiguate(frame, word, stopwordset=None, expand=True):
    # Ottengo l'insieme di tutti i synset disponibili per la parola
    # da disambiguare
    synsets = wn.synsets(word)
    # Il senso più frequente corrisponde al primo della lista
    best_sense = synsets[0]
    max_overlap = 0
    # Creo il contesto del Frame, utilizzando le funzioni
    # disponibili in FrameNet
    context = frame_ctx(frame, stopwordset)
    
    for sense in synsets:
        signature = get_signature(sense, stopwordset, expand)
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense
