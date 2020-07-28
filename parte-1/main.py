import nltk
from tree_translator import parse_s
from simplenlg import *

grammar = nltk.data.load('file:it_grammar.cfg')

frase = input('Scrivi una frase: ')

tokenized = nltk.word_tokenize(frase.lower(), language='italian')

parser = nltk.BottomUpChartParser(grammar)

lexicon = Lexicon.getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)

last_sentence = ''
for tree in parser.parse(tokenized):
    tree.draw()
    s = parse_s(tree, nlgFactory)
    frase = realiser.realiseSentence(s)
    if last_sentence != frase:
        print(frase)
        last_sentence = frase
