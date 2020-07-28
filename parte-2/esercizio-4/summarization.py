import pandas as pd
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from utils import weighted_overlap, vec_to_inverse_rank_dict, cached_property, isfloat

lemmatizer = WordNetLemmatizer()



class Document:
    def __init__(self, text, word_vectors, parse_title=True, remove_first_line=True, stopw=None, sig_size=20, title_weight=5, first_sentence_bonus=1.3, last_sentence_bonus=1.05):
        self.text = text
        self.title = ''
        self.word_vectors = word_vectors[0]
        self.vec_dim = word_vectors[1]
        self.sig_size = sig_size
        self.title_weight = title_weight
        self.first_sentence_bonus = first_sentence_bonus
        self.last_sentence_bonus = last_sentence_bonus
        
        if stopw == None:
            stopw = set(stopwords.words('english'))
        self.stopw = stopw
        
        # Rimuovo la prima linea che indica il ink
        lines = text.split('\n')
        if remove_first_line:
            lines = lines[1:]
        
        # Imposto il titolo del documento
        remove_untill_index = 1
        if parse_title:
            for line in lines:
                if line != '':
                    self.title = Paragraph(line, 0, parent=self)
                    break
                remove_untill_index += 1
        # Mi sposto subito dopo il titolo
        lines = lines[remove_untill_index:]
        self.text = '\n'.join(lines)
        paragraphs = self.text.split('\n\n')
        self.paragraphs = [Paragraph(x, i, parent=self) for i, x in enumerate(paragraphs)]
    
    @cached_property
    def signature(self):
        # Ottengo tutti i keysenses del documento
        keysenses = self.keysenses
        # Ottengo tutti i keysenses del titolo
        title_keysenses = self.title.keysenses
        sig = {}
        count = 0
        # Prendo tutti i sensi in ordine di importanza e li inserisco
        # nella signature in base alla dimensione definita
        for k in sorted(keysenses, key=lambda k: keysenses[k][0], reverse=True):
            if count >= self.sig_size:
                break
            sig[k] = keysenses[k]
            count+=1
        
        # Aggiungo alla signature le parole del titolo attribuendo un peso maggiore
        for synset_id, value in title_keysenses.items():
            count = value[0]
            vec = value[1]
            if not synset_id in sig:
                sig[synset_id] = (count * self.title_weight, vec)
            else:
                sig[synset_id] = (sig[synset_id][0] + count * self.title_weight, sig[synset_id][1])
        
        return sig
    
    def summary(self, compression_ratio=0.3, to_string=True):
        summary = []
        # Definisco il numero massimo di caratteri che rispettano il compression ratio
        max_chars = len(self.text) * compression_ratio
        cur_chars = 0
        # Partendo dalle frasi piu' importanti, comincio a comporre il riassunto
        # tenendo conto man mano della lunghezza
        for i, sent in sorted(enumerate(self.itersentences()), key=lambda x: x[1].sentence_score, reverse=True):
            if cur_chars + len(sent.text) > max_chars:
                break
            summary.append((i,sent))
            cur_chars += len(sent.text)
        
        # Riordino le frasi secondo l'ordine che avevano nel documento originale
        # (specificato dall'indice i)
        res = []
        for i, sent in sorted(summary):
            res.append(sent)
        
        # Trasformo la lista di frasi in una stringa contenente il riassunto
        if to_string:
            tmp = ''
            for s in res:
                tmp += s.text + '\n'
            res = tmp
        return res
    
    @cached_property
    def keywords(self):
        # Ottengo tutte le keywords dai paragrafi
        kws = {}
        for paragraph in self.paragraphs:
            kw = paragraph.keywords
            for k, v in kw.items():
                if not k in kws.keys():
                    kws[k] = v
                else:
                    kws[k] = kws[k] + v
        return kws
    
    @cached_property
    def keysenses(self):
        # Ottengo tutti i keysenses dai paragrafi
        kws = {}
        for paragraph in self.paragraphs:
            keysenses = paragraph.keysenses
            for k, v in keysenses.items():
                if not k in kws.keys():
                    kws[k] = v
                else:
                    kws[k] = (kws[k][0] + v[0], kws[k][1])
        return kws
    
    def set_vec_index(self, column=None):
        # Riordina il dataframe (nasari) in base ad una colonna e la imposta come indice
        # utile per rendere più veloce la ricerca di un elemento su tale colonna
        if column == None:
            self.word_vectors = self.word_vectors.reset_index()
        else:
            if self.word_vectors.index.name != column:
                self.word_vectors = self.word_vectors.reset_index()
                self.word_vectors = self.word_vectors.set_index(column).sort_index()
    
    def itersentences(self):
        # Generatore usato per iterare sulle frasi di questo documento
        for par in self.paragraphs:
            for sent in par.sentences:
                yield sent
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.text





class Paragraph:
    def __init__(self, text, index, parent):
        self.index = index
        self.text = text
        self.parent = parent
        sentences = sent_tokenize(text)
        self.sentences = [Sentence(x, i, parent=self) for i, x in enumerate(sentences)]
    
    @cached_property
    def keywords(self):
        # Ottengo tutte le keywords dalle frasi del paragrafo
        kws = {}
        for sentence in self.sentences:
            kw = sentence.keywords
            for k, v in kw.items():
                if not k in kws.keys():
                    kws[k] = v
                else:
                    kws[k] = kws[k] + v
        return kws
    
    @cached_property
    def keysenses(self):
        # Ottengo tutti i keysenses dalle frasi del paragrafo
        kws = {}
        for sentence in self.sentences:
            keysenses = sentence.keysenses
            for k, v in keysenses.items():
                if not k in kws.keys():
                    kws[k] = v
                else:
                    kws[k] = (kws[k][0] + v[0], kws[k][1])
        return kws
    
    @cached_property
    def calculate_cohesion_score(self):
        # Calcolo il cohesion score con gli altri paragrafi confrontando
        # tutti i keysenses ed utilizzando il weighted overlap
        self_ks = self.keysenses
        score = 0
        count = 0
        for par in self.parent.paragraphs:
            if par != self:
                other_ks = par.keysenses
                for self_synset, self_value in self_ks.items():
                    self_weight = self_value[0]
                    self_vec = self_value[1]
                    for other_synset, other_value in other_ks.items():
                        other_weight = other_value[0]
                        other_vec = other_value[1]
                        score += weighted_overlap(self_vec, other_vec, self.parent.vec_dim) * (self_weight + other_weight)
                        count += 1
        return score / count
        
    def iterwords(self):
        # Generatore usato per iterare sulle parole di questo paragrafo
        for sent in self.sentences:
            for words in sent.words:
                yield words
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f'Paragraph({self.text})'





class Sentence:
    def __init__(self, text, index, parent):
        self.index = index
        self.text = text
        self.parent = parent
        words = word_tokenize(text)
        self.words = [Word(x, i, parent=self) for i, x in enumerate(words)]
    
    @cached_property
    def keywords(self):
        # Creo un dizionario contenente tutte le parole che non sono stopwords
        # in forma lemmatizzata
        # chiave = parola lemmatizzata
        # valore = elenco di istanze di tutte le parole che hanno la stessa lemmatizzazione
        stopw = self.parent.parent.stopw
        words = [wd for wd in self.words if not wd.lemmatized in stopw and wd.text not in string.punctuation and not isfloat(wd.text)]
        word_dict = {}
        
        for word in words:
            if not word.lemmatized in word_dict.keys():
                word_dict[word.lemmatized] = [word]
            else:
                word_dict[word.lemmatized].append(word)
        
        return word_dict
    
    @cached_property
    def keysenses(self):
        # Recupero il senso disambiguato per ogni parola contenuta nel dizionario delle keywords
        # e creo un dizionario:
        # chiave = BabelSynsetID
        # valore = tupla con 2 valori:  
        # - numero di occorrenze del senso
        # - vettore corrispondente al BabelSynsetID
        kw = self.keywords
        senses_dict = {}
        for lemmatized, words in kw.items():
            for word in words:
                vec = word.nasari_vec
                if vec is not None:
                    if not str(vec.loc['BabelSynsetID']) in senses_dict.keys():
                        senses_dict[str(vec.loc['BabelSynsetID'])] = (1, vec)
                    else:
                        senses_dict[str(vec.loc['BabelSynsetID'])] = (1 + senses_dict[str(vec.loc['BabelSynsetID'])][0], senses_dict[str(vec.loc['BabelSynsetID'])][1])
        return senses_dict
                    
    
    @cached_property
    def calculate_topic_signature_agreement_score(self):
        # Recupero la signature del documento
        sig = self.parent.parent.signature
        # Recupero i sensi della frase corrente
        keysenses = self.keysenses
        score = 0
        # Per ogni coppia di sensi presi dai keysenses della frase e dalla signature
        # calcolo la weighted overlap e la sommo al punteggio
        for synset_id, value in keysenses.items():
            count = value[0]
            vec = value[1]
            for sig_vec, sig_val in sig.items():
                sig_weight = sig_val[0]
                sig_vec = sig_val[1]
                score += weighted_overlap(vec, sig_vec, self.parent.parent.vec_dim) * sig_weight
        return score
    
    @cached_property
    def sentence_score(self):
        # Assegno un punteggio alla frase considerando la sua posizione,
        # la coesione e il topic agreement
        first_bonus = self.parent.parent.first_sentence_bonus
        last_bonus = self.parent.parent.last_sentence_bonus
        bonus = 1
        if self.index == 0:
            bonus = first_bonus
        elif self.index == len(self.parent.sentences) - 1:
            bonus = last_bonus
        topic_agreement = self.calculate_topic_signature_agreement_score
        cohesion = self.parent.calculate_cohesion_score
        return topic_agreement * cohesion * bonus
        
    def sense_suitability_score(self, sense):
        # ritorna un punteggio usato per disambiguare una parola che ha come
        # contesto questa frase
        kw = self.keywords
        sense_dims = vec_to_inverse_rank_dict(sense, self.parent.parent.vec_dim)
        overlap = set(kw.keys()) & set(sense_dims.keys())
        score = 0
        for wd in overlap:
            score += len(kw[wd]) * sense_dims[wd]
        return score 
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f'Sentence({self.text})'





class Word:
    def __init__(self, text, index, parent):
        self.text = text
        self.index = index
        self.parent = parent
        self.lower = text.lower()
        self.lemmatized = lemmatizer.lemmatize(self.lower)
    
    @cached_property
    def nasari_vec(self):
        # metodo usato per disambiguare questa parola usando come contesto
        # la frase a cui la parola appartiene
        self.parent.parent.parent.set_vec_index(column='Name')
        vecs = self.parent.parent.parent.word_vectors
        
        try:
            senses = vecs.loc[self.lemmatized]
        except KeyError:
            return None

        if type(senses) == pd.Series:
            return senses
        best_score = 0
        best_sense = senses.iloc[0]
        for index, sense in senses.iterrows():
            score = self.parent.sense_suitability_score(sense)
            if score > best_score:
                best_score = score
                best_sense = sense
        return best_sense
    
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f'Word({self.text})'