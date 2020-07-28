import os, glob
import spacy
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from itertools import chain

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

def read_article(path):
    with open(path, 'r') as f:
        text = f.read()
    title = text.split('\n')[0]
    text = text[len(title):]
    return title, text.strip()

def read_all_articles_in_folder(path):
    path = os.path.join(path, '*.txt')
    for file in glob.glob(path):
        yield read_article(file)
        
def read_all_articles(paths):
    return chain(*[read_all_articles_in_folder(x) for x in paths])

def count_articles(paths):
    count = 0
    for path in paths:
        path = os.path.join(path, '*.txt')
        count += len(glob.glob(path))
    return count

def read_sentences(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences

def get_verb_args(doc, token):
    args = [c for c in token.children if c.dep_ in ('nsubj', 'dobj')]
    if len(args) == 2:
        if args[0].dep_ == 'nsubj' and args[1].dep_ == 'dobj':
            return args
        elif args[0].dep_ == 'dobj' and args[1].dep_ == 'nsubj':
            return [args[1], args[0]]
    return []

def ctx(sentence):
    return {x.lemma_ for x in sentence if not x.is_stop and not x.pos_ in ('SYM', 'NUM', 'PUNCT')}

def get_signature(sense, expand=True):
    # Per ottenere la signature splitto la frase in singole 
    # parole tramite word_tokenize e poi estraggo il contesto
    signature = ctx(nlp(sense.definition()))
    
    # Effettuo la stessa operazione per gli esempi del sysnet 
    # presenti in wordnet
    for example in sense.examples():
        signature = signature.union(ctx(nlp(example)))
    
    # Aggiungo i sinonimi lemmatizzati
    synonyms = {lemmatizer.lemmatize(x.lower()) for x in sense.lemma_names() if not '_' in x}
    signature = signature.union(synonyms)

    # Per migliorare la precisione dell'algoritmo aggiungo
    # inoltre gli iperonimi, gli iponimi e i meronimi alla
    # signature
    if expand:
        hypernym_sig, hyponym_sig, meronym_sig = set(), set(), set()
        
        for hypernym in sense.hypernyms() + sense.instance_hypernyms():
            hypernym_sig = hypernym_sig.union(get_signature(hypernym, False))
            
        for hyponym in sense.hyponyms() + sense.instance_hyponyms():
            hyponym_sig = hyponym_sig.union(get_signature(hyponym, False))
            
        for meronym in sense.part_meronyms() + sense.member_meronyms():
            meronym_sig = meronym_sig.union(get_signature(meronym, False))
        
        signature = signature.union(hypernym_sig)
        signature = signature.union(hyponym_sig)
        signature = signature.union(meronym_sig)
    
    return signature

def lesk_disambiguate(token, pos = wn.NOUN, expand=True):
    sentence = token.sent
    synsets = wn.synsets(token.lemma_, pos = pos)
    # Il senso più frequente corrisponde al primo della lista
    best_sense = synsets[0]
    max_overlap = 0
    # Estraggo il contesto della frase
    context = ctx(sentence)
    
    for sense in synsets:
        # Ottengo la signature del synset che sto esaminando
        signature = get_signature(sense, expand)
        # Confronto quanti elementi ci sono in comune tra la signature del
        # senso che sto esaminando e il contesto (la frase)
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense