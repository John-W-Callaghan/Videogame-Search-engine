#import necessary libraries and function
import os
from bs4 import BeautifulSoup , Comment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tokenisor import tokenizor
import numpy as np
from nltk.tokenize import word_tokenize
import string
import time
from nltk.corpus import wordnet

start_time = time.time()

#extract title content and url
def Htmlread(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        #Extract relevant information (modify as per your dataset)
        title = soup.title.text.strip()
        content = soup.get_text().strip()
        return {'title': title, 'content': content}
    

#Function to find the main paragraph in the text to output with the result
def description(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        start_comment = soup.find(string=lambda text: isinstance(text, Comment) and 'DESCRIPTION' in text)
        end_comment = soup.find(string=lambda text: isinstance(text, Comment) and '/DESCRIPTION' in text)

        if start_comment and end_comment:
            start_index = html_content.find(str(start_comment)) + len(str(start_comment))
            end_index = html_content.find(str(end_comment))
            description = html_content[start_index:end_index].strip()
            return description
        else:
            return None


#added after experimentation but is used for query expansion
def preprocess_query(query):
    tokens = word_tokenize(query)
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Expand query with synonyms
    expanded_tokens = []
    for token in tokens:
        expanded_tokens.append(token)
        synsets = wordnet.synsets(token)
        for synset in synsets:
            expanded_tokens.extend(synset.lemmas()[0].name().split('_'))
    
    return ' '.join(expanded_tokens)

#search function using tdf-if
def search(queries, tfidf_matrix, vectorizer, documents, top_n=10):
    results_for_queries = []
    for query in queries:
        expanded_query = preprocess_query(query)
        query_tfidf = vectorizer.transform([expanded_query])
        cosine_similarities = np.dot(query_tfidf, tfidf_matrix.T).toarray().flatten()
        top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
        results = [{'title': documents[idx]['title'],
                    'description': description(file_path),
                    'file_name': os.path.basename(documents[idx]['file_path'])} for idx, file_path in zip(top_indices, [documents[idx]['file_path'] for idx in top_indices])]
        results_for_queries.append(results)
    return results_for_queries




#reads html files and create dictionary for each
def processdata(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):
            file_path = os.path.join(folder_path, filename)
            file_data = Htmlread(file_path)
            file_data['file_path'] = file_path
            data.append(file_data)
    return data

#folder path and call processdata function to process each item in the folder
videogamesfolder = 'videogames'
extracted_data = processdata(videogamesfolder)



#Extract content from each dictionary in 'extracted_data' and preprocess
documents = [tokenizor(item['content']) for item in extracted_data]



#pre built tdf library
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)



#Take user input for search
user_queries = input("Please enter the game, genre, or publisher you are interested in?")
queries = user_queries.split(',')

# Strip whitespace from each query
queries = [query.strip() for query in queries]

#search using user input
search_results = search(queries, tfidf_matrix, vectorizer, extracted_data)



#for loop to print the title url and description
for i, results in enumerate(search_results):
    print(f"Results for query '{queries[i]}':")
    for result in results:
        #print("\n")
        print(f"{result['title']}")
        #print(f"URL:{result['file_name']}")
       # print(f"{result['description']}")
        #print("\n")
