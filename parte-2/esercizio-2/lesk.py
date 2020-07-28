from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import string

lemmatizer = WordNetLemmatizer()

def ctx(sentence, stopwordset=None):
    if stopwordset == None:
        stopwordset = set(stopwords.words('english'))
    
    # Estraggo il contesto effettuando la lemmatizzazione di tutte
    # le parole che non sono stopwords e/o segni di punteggiatura 
    context = {lemmatizer.lemmatize(x.lower()) for x in sentence if not x.lower() in stopwordset and not x in string.punctuation}
    return context

def get_signature(sense, stopwordset=None, expand=True):
    # Per ottenere la signature splitto la frase in singole 
    # parole tramite word_tokenize e poi estraggo il contesto
    signature = ctx(word_tokenize(sense.definition()), stopwordset)
    
    # Effettuo la stessa operazione per gli esempi del sysnet 
    # presenti in wordnet
    for example in sense.examples():
        signature = signature.union(ctx(word_tokenize(example), stopwordset))
    
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

def lesk_disambiguate(sentence, word, stopwordset=None, expand=True):
    if type(word) == int:
        word = sentence[word]
    synsets = wn.synsets(word)
    # Il senso più frequente corrisponde al primo della lista
    best_sense = synsets[0]
    max_overlap = 0
    # Estraggo il contesto della frase
    context = ctx(sentence, stopwordset)
    
    for sense in synsets:
        # Ottengo la signature del synset che sto esaminando
        signature = get_signature(sense, stopwordset, expand)
        # Confronto quanti elementi ci sono in comune tra la signature del
        # senso che sto esaminando e il contesto (la frase)
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense