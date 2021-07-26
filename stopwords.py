from nltk.corpus import stopwords
from nltk import download as download_stop_words
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

if stopwords:
    pass
else:
    download_stop_words('stopwords')

stopwords_russian = stopwords.words('russian')
stopwords_russian.pop(stopwords_russian.index('не'))
stopwords_russian.pop(stopwords_russian.index('нет'))
stopwords_russian.pop(stopwords_russian.index('нельзя'))
stopwords_russian.pop(stopwords_russian.index('лучше'))
stopwords_russian.pop(stopwords_russian.index('никогда'))
stopwords_russian.pop(stopwords_russian.index('ничего'))