# author: avrahami, py3 code
import re
from collections import defaultdict
import spacy
import pandas as pd
import textwrap
from nltk.tokenize import sent_tokenize
from relations_between_srs.Sentence import Sentence

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)


class SrRelations(object):

    def __init__(self, primary_sr, all_sr,
                 not_intresting_sr={'place', 'all', 'askreddit', 'rocketleagueexchange', 'globaloffensivetrade'}):
        if any([name.startswith('r/') for name in not_intresting_sr]):
            raise IOError("Parameters given as input are not valid")
        # if labels are continuous, then bin_num_labels should be int
        self.primary_sr = primary_sr
        self.all_sr = all_sr
        self.not_intresting_sr = not_intresting_sr
        self.subm_explicit_relations = defaultdict(set)
        self.comm_explicit_relations = defaultdict(set)
        self.sentences = []


    def find_explicit_relations(self, reddit_data, submission_data=True):
        # regrex which means: the 'r/' and then any char besides any of the: [\,:,#... and whitespace] 0-inf times
        regex = 'r/[^\\:#.,!?()\/\]\[\s]*'
        # regex = 'r/[^.,!?()\s]*'
        # making sure sr is not in the not_intresting_sr list
        redundent_sr = set([str(self.primary_sr.lower())] + [str(sr.lower()) for sr in self.not_intresting_sr])
        # looping over each of row, to find any r/[SUBREDDIT] instances (this is the most straight forward approach)
        for index, row in reddit_data.iterrows():
            cur_id = row["id"]
            if submission_data:
                cur_title = row.loc['title']#.encode('utf-8')
                cur_selftext = row.loc['selftext']#.encode('utf-8')
                sr_found = re.findall(regex, cur_title)
                sr_found += re.findall(regex, cur_selftext)
            else:
                cur_body = row.loc['body']#.encode('utf-8')
                sr_found = re.findall(regex, cur_body)
            sr_found = list(set(sr_found))
            # removing the r/ prefix
            sr_found_normalized = [sr[2:].lower() for sr in sr_found]
            # cleaning the list - only ones which are not redundant and exist in the full list of subreddits
            sr_found_normalized = [sr for sr in sr_found_normalized if sr not in redundent_sr and sr in self.all_sr]
            # for each of the sr we found, we'll add them to the dictionary of the submissions/comments
            for name in sr_found_normalized:
                if submission_data:
                    self.subm_explicit_relations[name].update({cur_id})
                else:
                    self.comm_explicit_relations[name].update({cur_id})

        # now, looping over each token and checking if it equals one of the tokens ew are looking for (i.e., sr)
        nlp = spacy.load('en')
        for index, row in reddit_data.iterrows():
            cur_id = row["id"]
            if submission_data:
                cur_title = row.loc['title']#.encode('utf-8')
                cur_selftext = row.loc['selftext']#.encode('utf-8')
                # merging together the header and the selftext
                cur_full_text = cur_title + '. ' + cur_selftext
                doc_text = nlp(cur_full_text)
                # looping over each token in the current reddit submission
                for token in doc_text:
                    token_normalized = token.text.lower()
                    # case the token is a punctuation one
                    if token.is_punct:
                        continue
                    elif token_normalized in self.all_sr and token_normalized not in redundent_sr:
                        self.subm_explicit_relations[token.text.lower()].update({cur_id})
                        continue
                    elif token.lemma_ in self.all_sr and token.lemma_ not in redundent_sr:
                        self.subm_explicit_relations[token.text.lower()].update({cur_id})
            # case we are dealing with the comments data
            else:
                cur_body = row.loc['body']
                doc_text = nlp(cur_body)
                # looping over each token in the current reddit submission
                for token in doc_text:
                    token_normalized = token.text.lower()
                    if token.is_punct:
                        continue
                    elif token_normalized in self.all_sr and token_normalized not in redundent_sr:
                        self.comm_explicit_relations[token.text.lower()].update({cur_id})
                        continue
                    elif token.lemma_ in self.all_sr and token.lemma_ not in redundent_sr:
                        self.comm_explicit_relations[token.lemma_].update({cur_id})

    def print_findings(self, reddit_data, file_name, submission_data=True):
        with open(file_name, "a") as text_file:
            # creating a combined dict of all the submissions/comments (explicit and implicit occurrences)
            if submission_data:
                relevant_dict = self.subm_explicit_relations.copy()
            else:
                relevant_dict = self.comm_explicit_relations.copy()
            # printing all the explicit ones
            for ref_sr, ids in relevant_dict.items():
                if submission_data:
                    print("\n\nMentions in the submissions related to: {}".format(ref_sr), file=text_file)
                else:
                    print("\n\nMentions in the comments related to: {}".format(ref_sr), file=text_file)
                ids = set(ids)  # in order to remove duplications
                for id in ids:
                    cur_data = reddit_data[reddit_data["id"] == id]
                    if submission_data:
                        text_to_file = cur_data['title'].values[0] + " ******* " + cur_data['selftext'].values[0]
                    else:
                        text_to_file = cur_data['body'].values[0]
                    textwrap.dedent(text_to_file)
                    print("\treddit_id: {}. full text:{}".format(id, text_to_file.encode('utf-8')), file=text_file)

    def information_extraction(self, reddit_data, submission_data=True, only_explicit_relations=False):
        # creating a combined dict of all the submissions/comments (explicit and implicit occurrences)
        if submission_data and only_explicit_relations:
            relevant_dict = self.subm_explicit_relations.copy()
        elif not submission_data and only_explicit_relations:
            relevant_dict = self.comm_explicit_relations.copy()
        # case only_explicit_relations is False, which means we are running the code over all reddit data given in input
        # in such case, the dict is a dummy one, all ids are mapped to the same key (called "general")
        else:
            relevant_dict = {"general": set(reddit_data["id"])}
        # pulling out all the text relevant to current use-case (this loop runs over each subreddit found
        redundent_sr = set([str(self.primary_sr.lower())] + [str(sr.lower()) for sr in self.not_intresting_sr])
        nlp = spacy.load('en_core_web_sm')
        for ref_sr, ids in relevant_dict.items():
            # this loop runs over each submission/comment id, which is in the ids dictionary
            for cur_id in ids:
                cur_data = reddit_data[reddit_data["id"] == cur_id]
                if submission_data:
                    title = cur_data['title'].values[0]
                    tokenize_list = sent_tokenize(title)
                    selftext = cur_data['selftext'].values[0]
                    # if we see a special case when the submission is a deleted/removed/empty one, we'll skip next step
                    if selftext not in ['[deleted]', '[removed]', '']:
                        tokenize_list = tokenize_list + sent_tokenize(selftext)
                else:
                    body = cur_data['body'].values[0]
                    tokenize_list = sent_tokenize(body)
                # now we have tokenize_list (which is a list of sentences) and we need to find useful information in it
                for sent_idx, sent in enumerate(tokenize_list):
                    #spacy_sentence = nlp(sent)
                    #self._add_ner_markers(spacy_sentence=spacy_sentence,
                    #                      file_path=self.primary_sr + '_sentences_with_NERs.txt')
                    # dependency parsing analysis + NER
                    sent_obj = Sentence(sr=self.primary_sr, id=cur_id, sent_idx=sent_idx, sent_text=sent, lang='en')
                    sent_obj.analyse_sent(relevant_sr_list=self.all_sr - redundent_sr, nlp=nlp)
                    self.sentences.append(sent_obj)

    def sentences_summary(self):
        summary_df = pd.DataFrame(columns=['sr', 'id', 'sent_idx', 'sent_text', 'other_sr_mentioned',
                                           'verbs', 'sent_root', 'leading_verbs'])
        for cur_sent_obj in self.sentences:
            summary_df = summary_df.append(cur_sent_obj.sent_summary(), ignore_index=True)
        return summary_df


    def _add_ner_markers(self, spacy_sentence, file_path, mark_sr=True):
        # changing the sentence so NERs occurrences will be marked
        modified_text = spacy_sentence.text
        added_chars = 0
        valid_ents = 0
        # opening the file in which we are pushing the modified data into
        with open(file_path, "a") as text_file:
            for ent in spacy_sentence.ents:
                if len(ent.text) == 0 or len(ent.text.strip()) == 0:
                    continue
                prefix = '<' + ent.label_ + '>'
                suffix = '</' + ent.label_ + '>'
                # case r/ or /r/ is inside the entity found by spacy (e.g. entity is 'r/canada')
                if 'r/' in ent.text.lower() or '/r' in ent.text.lower():
                    continue
                # case the entity found is exactly the subreddit name, without the r/ ot /r/ (e.g. entity is canada, but
                # before this word appears '/r')
                if (ent.start_char >= 2 and modified_text[ent.start_char-2:ent.start_char] == 'r/') or \
                        (ent.start_char >= 3 and modified_text[ent.start_char - 3:ent.start_char] == '/r/'):
                    continue
                # markering the entity with <LOC> ... </LOC> for example
                modified_text = modified_text[:ent.start_char + added_chars] + prefix +\
                                modified_text[ent.start_char + added_chars:ent.end_char + added_chars] + \
                                suffix + modified_text[ent.end_char + added_chars:]
                added_chars += len(prefix) + len(suffix)
                valid_ents += 1

            # case we wish to also mark sub-reddits as entities. This will be a second loop over the text
            if mark_sr:
                text_before_sr_changes = modified_text
                prefix = '<' + 'SR' + '>'
                suffix = '</' + 'SR' + '>'
                # this regex means and word that starts with r/ and any word that start with /r/
                regex_list = ["r/[^\s\]\[\/(){}~`!@#$%^&*,.?><]*", "/r/[^\s\]\[\/(){}~`!@#$%^&*,.?><]*",
                              "R/[^\s\]\[\/(){}~`!@#$%^&*,.?><]*", "/R/[^\s\]\[\/(){}~`!@#$%^&*,.?><]*"]
                regex = "|".join(regex_list)
                p = re.compile(regex)
                added_chars = 0
                # looping over all the sr instances found be the regex compiler and updating them
                for sr in p.finditer(text_before_sr_changes):
                    starting_pos = sr.start()
                    ending_pos = sr.end()
                    modified_text = modified_text[:starting_pos + added_chars] + prefix +\
                                    modified_text[starting_pos + added_chars:ending_pos + added_chars] + \
                                    suffix + modified_text[ending_pos + added_chars:]
                    added_chars += len(prefix) + len(suffix)
            print("{}".format(modified_text).encode('utf-8'), file=text_file)
            print("Current sentence after adding markers for NERs is: {}".format(modified_text))
