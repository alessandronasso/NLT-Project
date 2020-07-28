from simplenlg import Tense, NumberAgreement
# Per verbi (Traduzione_EN, 'V', Subj, Tempo, Perfect, Passive)
# Per nomi (Traduzione_EN, 'N', Plural)
# Per altro (Traduzione_EN, 'A')

it_en_dict = {
    'è': ('be', 'V', 'it', Tense.PRESENT, False, False),
    'è-stato-sconfitto': ('defeat', 'V', 'he', Tense.PRESENT, True, True),
    'sono-stati-spazzati': ('sweep', 'V', 'they', Tense.PRESENT, True, True),
    'sono': ('be', 'V', 'they', Tense.PRESENT, False, False),
    'ha': ('have', 'V', 'he', Tense.PRESENT, False, False),
    'ha-fatto': ('make', 'V', 'he', Tense.PAST, False, False),
    'ha-spazzato': ('sweep', 'V', 'he', Tense.PAST, False, False),
    'ha-sconfitto': ('defeat', 'V', 'he', Tense.PAST, False, False),
    'fatto': ('fact', 'N', NumberAgreement.SINGULAR),
    'spada': ('sword', 'N', NumberAgreement.SINGULAR),
    'laser': ('laser', 'N', NumberAgreement.SINGULAR),
    'sword-laser': ('lightsaber', 'N', NumberAgreement.SINGULAR),
    'padre': ('father', 'N', NumberAgreement.SINGULAR),
    'mossa': ('move', 'N', NumberAgreement.SINGULAR),
    'avanzi': ('remnant', 'N', NumberAgreement.PLURAL),
    'repubblica': ('Republic', 'N', NumberAgreement.SINGULAR),
    'stati': ('state', 'N', NumberAgreement.PLURAL),
    'la': ('the', 'A'),
    'una': ('a', 'A'),
    'gli': ('the', 'A'),
    'tuo': ('your', 'A'),
    'di': ('of', 'A'),
    'della': ('of the', 'A'),
    'leale': ('fair', 'A'),
    'ultimi': ('last', 'A'),
    'vecchia': ('old', 'A'),
    'via': ('away', 'A')
}
