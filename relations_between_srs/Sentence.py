import spacy
import re
import pandas as pd
import gc


class Sentence(object):
    def __init__(self, sr, id, sent_idx, sent_text, lang='en'):
        self.sr = sr
        self.id = id
        self.sent_idx = sent_idx
        self.sent_text = sent_text
        self.lang = lang
        self.other_sr_mentioned = set()
        self.ners = []
        self.verbs = []
        self.sent_root = None
        self.leading_verbs = []
        self.details_per_word = pd.DataFrame(columns=['index', 'text', 'text_pos', 'head_text', 'head_pos',
                                                      'dep_type', 'children'])

    def analyse_sent(self, relevant_sr_list, nlp):
        #if self.lang == 'en':
        #    nlp = spacy.load('en_core_web_sm')
        # finding relevent srs in the text
        regex_list = ["r/[^\s\]\[\/(){}~`!@#$%^&*,.?><]*", "R/[^\s\]\[\/(){}~`!@#$%^&*,.?><]*"]
        regex = "|".join(regex_list)
        sr_found = re.findall(regex, self.sent_text)
        sr_found_normalized = [sr[2:].lower() for sr in sr_found]
        sr_found_normalized = [sr for sr in sr_found_normalized if sr in relevant_sr_list]
        self.other_sr_mentioned = set(sr_found_normalized)
        spacy_sentence = nlp(self.sent_text)
        # adding all the ners found
        for ent in spacy_sentence.ents:
            # case the ent found is not one which interests us
            if ent.label_ not in ['NORP', 'GPE', 'LOC']:
                continue
            # case the ent is empty - not interesting
            if len(ent.text) == 0 or len(ent.text.strip()) == 0:
                continue
            # case the ent is a sr we already marked before - not interesting
            if 'r/' in ent.text.lower() or '/r' in ent.text.lower():
                continue
            # case the entity found is exactly the subreddit name, without the r/ ot /r/ (e.g. entity is canada, but
            # before this word appears '/r')
            if (ent.start_char >= 2 and spacy_sentence.text[ent.start_char - 2:ent.start_char] == 'r/') or \
                    (ent.start_char >= 3 and spacy_sentence.text[ent.start_char - 3:ent.start_char] == '/r/'):
                continue
            # case it does interest us - we'll add it to the list
            self.ners.append((ent, ent.label_))
        # adding all the verbs found, the sentence root and the "leading verb"
        for index, token in enumerate(spacy_sentence):
            if token.is_punct:
                continue
            # if the token is a verb, it will be added to our list of verbs
            if token.pos_ == 'VERB':
                self.verbs.append((token, token.lemma_))
            # if the token is the root, it will be added to our object as the root of the sentence
            if token.dep_ == 'ROOT':
                self.sent_root = (token, token.lemma_)
                # case is the root and the verb (preferred case)
                if token.pos_ == 'VERB':
                    self.leading_verbs.append((token, token.lemma_))
                # case the root is not a verb, need to look for the leading verb in his children
                else:
                    for possible_verb in token.children:
                        if possible_verb.pos == 'VERB':
                            self.leading_verbs.append((possible_verb, possible_verb.lemma_))
        # end of the for loop, now the object should be "filled" with all information
        self.details_per_word.append({'index': index, 'text': token.text, 'text_pos': token.pos_,
                                      'head_text': token.head.text, 'head_pos': token.head.pos_,
                                      'dep_type': token.dep_,
                                      'children': str([child for child in token.children
                                                       if not child.is_punct])},
                                     ignore_index=True)

    def sent_summary(self):
        result = pd.Series({'sr': self.sr, 'id': self.id, 'sent_idx': self.sent_idx, 'sent_text': self.sent_text,
                            'other_sr_mentioned': self.other_sr_mentioned, 'ners': self.ners,
                            'verbs': self.verbs, 'sent_root': self.sent_root, 'leading_verbs': self.leading_verbs})
        return result
