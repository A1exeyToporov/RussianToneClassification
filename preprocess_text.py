from nltk.tokenize import TweetTokenizer
import re
import pymorphy2
from stopwords import stopwords_russian
from tqdm import tqdm

punctuation = '#$%&\()*+,-./:;<=>@[\\]^_`{|}~'

morph = pymorphy2.MorphAnalyzer()

def clean_text(text, normalize_form = False):
    """Clean text from punctuation, stop words. If normalize_form = True,
    function will normalize each word to it's initial form"""

    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'#', '', text)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    text_tokens = list(filter(lambda x: x not in stopwords_russian, text_tokens))
    text_tokens = list(filter(lambda x: x not in punctuation, text_tokens))
    if normalize_form:
      text_tokens = [morph.parse(i)[0].normal_form for i in text_tokens]
    text_tokens: str = ' '.join(text_tokens)
    text_tokens.encode('utf-8')
    return text_tokens

def clean(text_corpus, normalize_form = False):
    print ("Text cleaning in progress")
    text_corpus = [clean_text(x, normalize_form=normalize_form) for x in tqdm(text_corpus)]
    # text_corpus = text_corpus.apply(lambda x: clean_text(x, normalize_form=normalize_form))
    print("Done")
    return text_corpus
