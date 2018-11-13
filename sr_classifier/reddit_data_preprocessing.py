# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 9.10.2018

from spacy.tokenizer import Tokenizer
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string
import textwrap
from urllib.parse import urlparse
from collections import defaultdict
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "..", "...", "....", ".....", "......" "”", "”"]


class RedditDataPrep(object):
    """
    Class to run some data preperation phsases on reddit datasets
    it is mainly connected to removal of rows, marking text and tokanization and less related to cleaning of text.
    Such cleanning of text is done by the 'CleanTextTransformer' object

    Parameters
    ----------
    is_submission_data: bool
        whether this data-prep object will be applied on submission data (or comments data)
    lang: str, default: 'en'
        the language used in the sub-reddit. This should be a shortcut same as used by spacy
    include_deleted: bool, default: True
        whether deleted items should be part of the population. It is important since in submission the header of
        deleted posts is still public
    most_have_regex: regex expression or None, default: r'/\s*r\s*/\s*place|r\s*/\s*place'
        regex which must appear in the text in order fot the specific row to be included in the returned data. It
        is useful in cases when we want to filter our (mainly submission) which do not hold relevant information for us
    most_not_have_regex: regex expression or None, default: None
        regex which must not appear in the text in order fot the specific row to be included in the returned data
    remove_stop_words: bool, default: True
        whether removal of stop words along tokanization process should take place. List of stop-words appear in the top
        of this file
    remove_symbols: bool, default: True
        whether removal of symbols along tokanization process should take place. List of relevant symbols appear in
        the top of this file

    Attributes
    ----------
    same as parameters
    """

    def __init__(self, is_submission_data, lang='en', include_deleted=True,
                 most_have_regex=r'/\s*r\s*/\s*place|r\s*/\s*place', most_not_have_regex=None,
                 remove_stop_words=True, remove_symbols=True):
        self.is_submission_data = is_submission_data
        self.lang = lang
        self.include_deleted = include_deleted
        self.most_have_regex = most_have_regex
        self.most_not_have_regex = most_not_have_regex
        self.remove_stop_words = remove_stop_words
        self.remove_symbols = remove_symbols

    def data_pre_process(self, reddit_df):
        """
        Pre processes steps to the reddit data. Along these phases, no change to the text is made, only removals of
        non relevant rows (row = submission/comment) and the function returns a list of texts which can be eaaily used
        for later phases
        :param reddit_df: pandas df
            the dataframe with all submission/comments relevant to run the data prep on
        :return: tuple
            first element in the tuple is a pandas data-frame which is the original df, but filtered (rows perspective)
            second element is a list of tuples. Each tuple holds the submission/comment score, submission/comment header
            and the submission/comment self-text (since comments have no self-text, their 3rd item will be an empty
            string). Example of such tuple: (13, 'amazing work guys', '')
        """
        full_text = []
        # these are rows which were removed along the process (because of the 'most_have_regex' or most_not_have_regex)
        non_legit_idx = []
        if self.is_submission_data:
            for index, row in reddit_df.iterrows():
                # checking if the current row is relevant to our analysis (based on the regex logic)
                if not self._is_legit_row(row):
                    non_legit_idx.append(index)
                    continue
                self_text = row.loc['selftext']
                # case the self text is not empty and was not removed
                if self_text != '[removed]' and self_text != '[deleted]':
                    full_text += [(row.loc['score'], row.loc['title'], self_text)]
                    continue
                # case the content was removed but we want to include it in the data we generate
                if (self_text == '[removed]' or self_text == '[deleted]') and self.include_deleted:
                    full_text += [(row.loc['score'], row.loc['title'], '')]
        elif ~self.is_submission_data:
            for index, row in reddit_df.iterrows():
                body = row.loc['body']
                # case the self text is not empty and was not removed
                if body != '[deleted]' and body != '[removed]':
                    full_text += [(row.loc['score'], body, '')]
        legit_idx = set(reddit_df.index) - set(non_legit_idx)
        return reddit_df[reddit_df.index.isin(legit_idx)], full_text

    def _is_legit_row(self, row):
        """
        checking if a row is OK to be used, based on a regex value of the object
        :param row: pandas df
            a single pandas dataframe (one row) with at least the the columns 'title' and 'selftext' for submission
            object and 'body' for comment object
        :return: bool
            is this a legitimate case (row) to be used

        Examples
        ---------
        title_text = 'come help us in /r/place now!!'
        selftext = 'it is going to be fun. fuc it'
        data_prep_obj = RedditDataPrep(is_submission_data=True, most_have_regex=r'/\s*r\s*/\s*place|r\s*/\s*place',
        most_not_have_regex=r'fuck')
        row = pd.Series(data = {'title': title_text, 'selftext': selftext})
        data_prep_obj._is_legit_row(row=row)
        """
        # case both most-have and most-not-have are None - all rows are legit to be used
        if self.most_have_regex is None and self.most_not_have_regex is None:
            return True
        # case only the most-have is not None
        elif self.most_have_regex is not None and self.most_not_have_regex is None:
            regexp = re.compile(self.most_have_regex)
            if self.is_submission_data:
                if regexp.search(str(row.loc['title']).lower()) or regexp.search(str(row.loc['selftext']).lower()):
                    return True
                else:
                    return False
            # case we deal with comments data
            else:
                if regexp.search(row.loc['body']):
                    return True
                else:
                    return False
        # case only the most-not-have is not None
        elif self.most_have_regex is None and self.most_not_have_regex is not None:
            regexp = re.compile(self.most_not_have_regex)
            if self.is_submission_data:
                if regexp.search(row.loc['title'].lower()) or regexp.search(row.loc['selftext'].lower()):
                    return False
                else:
                    return True
            # case we deal with comments data
            else:
                if regexp.search(row.loc['body']):
                    return False
                else:
                    return True
        # case both are not None (the must and most-not regrex expressions)
        else:
            pos_regexp = re.compile(self.most_have_regex)
            neg_regexp = re.compile(self.most_not_have_regex)
            if self.is_submission_data:
                if (pos_regexp.search(row.loc['title'].lower()) or pos_regexp.search(row.loc['selftext'].lower()))\
                        and not bool(neg_regexp.search(row.loc['title'].lower()))\
                        and not bool(neg_regexp.search(row.loc['selftext'].lower())):
                    return True
                else:
                    return False
            # case we deal with comments data
            else:
                if pos_regexp.search(row.loc['body']) and not bool(neg_regexp.search(row.loc['body'].lower())):
                    return True
                else:
                    return False

    def tokenize_text(self, sample):
        """
        overriding the default spacy tokenizer, with one more relevant to reddit cases. In addition, pulling out the
        lemmas of each word and removing stop-words if needed
        :param sample: str
            the text string to tokenize
        :return: list
            list of tokens
        """

        nlp.tokenizer = self.create_reddit_tokenizer()
        sample_as_list = self._break_long_text(sample=sample, chunks_size=500000)
        full_tokens_list = []
        for cur_sample in sample_as_list:
            cur_tokens = nlp(cur_sample)
            cur_lemmas = []
            for tok in cur_tokens:
                cur_lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
            cur_tokens = cur_lemmas
            if self.remove_stop_words:
                cur_tokens = [tok for tok in cur_tokens if tok not in STOPLIST]
            if self.remove_symbols:
                cur_tokens = [tok for tok in cur_tokens if tok not in SYMBOLS]
            full_tokens_list += cur_tokens
        return full_tokens_list

    @classmethod
    def mark_urls(cls, text, marking_method='replace'):
        """
        finds and mark all the URLs within a text
        :param text: str
            the text string to mark
        :param marking_method: str
            either 'replace' or 'add'. 'replace' removes the url found and replaces it with the 'netloc' definition
            (by the urlparse). 'add' will add a '<link>' before the like found and a '</link>' right after
        :return: tuple
            first item in the tuple is the revised_text - which is the original text with the url markers
            second item in the tuple is the urls_found, which is a dict with counters for each 'netloc' key in the text
        """
        web_url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
        regex = re.compile(web_url_regex, re.IGNORECASE)
        revised_text = []
        loc_in_orig_str = 0
        urls_found = defaultdict(int)
        # looping over each URL found
        for idx, match in enumerate(regex.finditer(text)):
            full_url = match.group(0)
            starting_position = match.start()
            ending_position = match.end()
            url_parts = urlparse(full_url)
            urls_found[url_parts.netloc] += 1
            # case we want to remove the original link letters and replace it with the general website name of the link
            if marking_method == 'replace':
                revised_text += text[loc_in_orig_str:starting_position]
                revised_text += url_parts.netloc
                loc_in_orig_str = ending_position
            # case we want to add a <link> before the url and </link> at the end
            elif marking_method == 'add':
                revised_text += text[loc_in_orig_str:starting_position]
                revised_text += '<link>'
                revised_text += text[starting_position:ending_position]
                revised_text += '</link>'
                loc_in_orig_str = ending_position
            # case it is none of the above, no change will be made
            else:
                revised_text += text[loc_in_orig_str:starting_position]
                loc_in_orig_str = ending_position
        # at the end we need to add the postfix of the original string
        revised_text += text[loc_in_orig_str:]
        # converting it to a str object
        revised_text = "".join(revised_text)
        return revised_text, urls_found

    @classmethod
    def mark_coordinates(cls, text, marking_method='replace'):
        """
        finds and mark all the coordinates within a text (e.g. (200, 300) or [34,  55])
        :param text: str
            the text string to mark
        :param marking_method: str
            either 'replace' or 'add'. 'replace' removes the coordinates found and replaces it with a '(X_Y)' text
            'add' will add a '<coordinates>' before the coordinates found and a '</coordinates>' right after
        :return: tuple
            first item in the tuple is the revised_text - which is the original text with the url markers
            second item in the tuple is the coordinates_found, which is an int with number of coordinates found in text
        """
        # regex which represents and symbol like (800, 700) or 22 , 58
        coordinates_regex = r'[\(\[]*[0-9]+\s*[,*-]\s*[0-9]+[\)\]]*'
        regex = re.compile(coordinates_regex, re.IGNORECASE)
        revised_text = []
        loc_in_orig_str = 0
        coordinates_found = 0
        # looping over each URL found
        for idx, match in enumerate(regex.finditer(text)):
            coordinates_found += 1
            starting_position = match.start()
            ending_position = match.end()
            # case we want to remove the original link letters and replace it with the general website name of the link
            if marking_method == 'replace':
                revised_text += text[loc_in_orig_str:starting_position]
                revised_text += '(X_Y)'
                loc_in_orig_str = ending_position
            # case we want to add a <link> before the url and </link> at the end
            elif marking_method == 'add':
                revised_text += text[loc_in_orig_str:starting_position]
                revised_text += '<coordinates>'
                revised_text += text[starting_position:ending_position]
                revised_text += '</coordinates>'
                loc_in_orig_str = ending_position
            # case it is none of the above, no change will be made
            else:
                return text
        # at the end we need to add the postfix of the original string
        revised_text += text[loc_in_orig_str:]
        # converting it to a str object
        revised_text = "".join(revised_text)
        return revised_text, coordinates_found

    @classmethod
    def create_reddit_tokenizer(cls):
        """
        special case which we want to recognize:
        first two are for cases like 'r/place' or /r/place.
        third one is for coordinates like (807, 707) (which are in a heavy usage in the r/place dataset). This once can
        have many white spaces in-between the numbers

        Testing
        --------
        #sample =  'r/place needs some Tlou art! ..r/place.. needs some Tlou art..Can we get Ellie somewhere on r/place?..'
        sample = 'Help us in (720,660)! we need your help in r/place!'
        nlp = spacy.load('en')
        nlp.tokenizer = create_reddit_tokenizer(nlp)
        tokens = nlp(sample)
        print([(idx, w) for idx, w, in enumerate(tokens)])
        """
        my_prefix = ['r/[a-zA-Z0-9_]+', '/r/[a-zA-Z0-9_]+', '\(\s*[0-9]+\s*\,\s*[0-9]+\s*\)']

        all_prefixes_re = spacy.util.compile_prefix_regex(tuple(my_prefix + list(nlp.Defaults.prefixes)))

        infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes)))

        my_suffix = ['r/[a-zA-Z0-9_]+', '/r/[a-zA-Z0-9_]+', '\(\s*[0-9]+\s*\,\s*[0-9]+\s*\)']
        suffix_re = spacy.util.compile_suffix_regex(tuple(my_suffix + list(nlp.Defaults.suffixes)))

        # reddit_special_cases = re.compile(r'\([0-9]+\,[0-9]+\)')
        return Tokenizer(nlp.vocab, nlp.Defaults.tokenizer_exceptions,
                         prefix_search=all_prefixes_re.search,
                         infix_finditer=infix_re.finditer, suffix_search=suffix_re.search,
                         token_match=None)#reddit_special_cases.match)

    @classmethod
    def _break_long_text(cls, sample, chunks_size):
        if len(sample) < chunks_size:
            return [sample]
        else:
            return textwrap.wrap(text=sample, width=chunks_size)

