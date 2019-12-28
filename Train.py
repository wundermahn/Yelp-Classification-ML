# Import necessarily libraries
import nltk, re, pandas as pd
from nltk.corpus import stopwords
import sklearn, string, numpy as np, time, pickle
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

# This function removes numbers from an array
def remove_nums(arr): 
    # Declare a regular expression
    pattern = '[0-9]'  
    # Remove the pattern, which is a number
    arr = [re.sub(pattern, '', i) for i in arr]    
    # Return the array with numbers removed
    return arr

# This function cleans the passed in paragraph and parses it
def get_words(para):   
    # Create a set of stop words
    stop_words = set(stopwords.words('english'))
    # Split it into lower case    
    lower = para.lower().split()
    # Remove punctuation
    no_punctuation = (nopunc.translate(str.maketrans('', '', string.punctuation)) for nopunc in lower)
    # Remove integers
    no_integers = remove_nums(no_punctuation)
    # Remove stop words
    dirty_tokens = (data for data in no_integers if data not in stop_words)
    # Ensure it is not empty
    tokens = [data for data in dirty_tokens if data.strip()]
    # Ensure there is more than 1 character to make up the word
    tokens = [data for data in tokens if len(data) > 1]
    
    # Return the tokens
    return tokens 

# Perform a min max scaling technique
# Optimize the data for a neural network
def minmaxscale(data):
    # Create the scaler
    scaler = MinMaxScaler()
    # Scale the data passed in
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    # Return the scaled data
    return df_scaled

def main():
    # Load in the dev file
    tsv_file = "C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\train.tsv"
    
    # Create the data structure
    csv_table=pd.read_csv(tsv_file, sep='\t', header=None)
    csv_table.columns = ['class', 'ID', 'text']

    # Create the vocabulary via the tokenization process in get_words
    s = pd.Series(csv_table['text'])
    new = s.str.cat(sep=' ')
    vocab = get_words(new)

     # Create the overall corpus
    s = pd.Series(csv_table['text'])
    corpus = s.apply(lambda s: ' '.join(get_words(s)))

    # Create "dirty" and "clean" metrics
    csv_table['dirty'] = csv_table['text'].str.split().apply(len)
    csv_table['clean'] = csv_table['text'].apply(lambda s: len(get_words(s)))

    # Create a vectorizer
    vectorizer = TfidfVectorizer()
    # Compute tfidf values
    # This also updates the vectorizer
    test = vectorizer.fit_transform(corpus)

    # Create a dataframe from the vectorization procedure
    df = pd.DataFrame(data=test.todense(), columns=vectorizer.get_feature_names())
    
    # Merge results into final dataframe
    result = pd.concat([csv_table, df], axis=1, sort=False)

    # Create the classification labels
    Y = result['class']

    # Drop unnecessary fields
    result = result.drop('text', axis=1)
    result = result.drop('ID', axis=1)
    result = result.drop('class', axis=1)

    # Create a variable to hold the dataset
    X = result

    # Scale the dataset
    scaled = minmaxscale(X)

    # Create the classifiers
    mlp = MLPClassifier()
    rf = RandomForestClassifier()

    # Including the optimized one
    mlp_opt = MLPClassifier(
        activation = 'tanh',
        hidden_layer_sizes = (1000,),
        alpha = 0.009,
        learning_rate = 'adaptive',
        learning_rate_init = 0.01,
        max_iter = 250,
        momentum = 0.9,
        solver = 'lbfgs',
        warm_start = False
    )    

    # Train the classifiers
    print("Training Classifiers")
    mlp_opt.fit(scaled, Y)
    mlp.fit(scaled, Y)
    rf.fit(scaled, Y)

    # Write the models and vectorizer to disk
    dump(mlp_opt, "C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\Models\\mlp_opt.joblib")
    dump(mlp, "C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\Models\\mlp.joblib")
    dump(rf, "C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\Models\\rf.joblib")
    pickle.dump(vectorizer, open("C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\tfidf_vectorizer.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Trained Classifiers")

main()
