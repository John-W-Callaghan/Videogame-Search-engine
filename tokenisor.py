import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
# Use the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# tokenisor function
def tokenizor(text):
    #remove whitespace
    text = re.sub('\s+', ' ', text).strip()

    #regex to remove punctuation
    text_no_punctuation = re.sub(r'[^\w\s]', '', text)

    #tokenization and lower case everything
    tokens = word_tokenize(text_no_punctuation.lower())

    #use lemmatization 
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    #remove numbers and non-alphabet characters
    lemmatized_tokens = [token for token in lemmatized_tokens if token.isalpha()]

    #remove stopwords
    lemmatized_tokens = [token for token in lemmatized_tokens if token not in ENGLISH_STOP_WORDS]

    #create a list of sentences and check for titles
    sentences = sent_tokenize(text)
    titlesandheadings = [word_tokenize(sentence)[0] for sentence in sentences if sentence.endswith('h')]

    #will give titles and headings extra weighting
    titlesheadingsrep = [item for item in titlesandheadings for _ in range(2)]

    #add all tokens together
    all_tokens = lemmatized_tokens + titlesheadingsrep

    return " ".join(all_tokens)
