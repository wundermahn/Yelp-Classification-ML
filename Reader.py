# Import necessary libraries
from nltk.corpus import stopwords
import sklearn, string, nltk, re, pandas as pd, numpy, time
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pickle

# Function to collect required F1, Precision, and Recall Metrics
def collect_metrics(actuals, preds):
    # Create a confusion matrix
    matr = confusion_matrix(actuals, preds, labels=[2, 4])
    # Retrieve TN, FP, FN, and TP from the matrix
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(actuals, preds).ravel()

    # Compute precision
    precision = true_positive / (true_positive + false_positive)
    # Compute recall
    recall = true_positive / (true_positive + false_negative)
    # Compute F1
    f1 = 2*((precision*recall)/(precision + recall))

    # Return results
    return precision, recall, f1

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

    # Load the vectorizer logic used for the training set
    tfidf_vectorizer = pickle.load(open("C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\tfidf_vectorizer.pkl", 'rb'))

    # Load in the dev file
    tsv_file = "C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\test.tsv"
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

    # Compute tfidf values
    tfidf = tfidf_vectorizer.transform(corpus)

    # Create a dataframe from the vectorization procedure
    df = pd.DataFrame(data=tfidf.todense(), columns=tfidf_vectorizer.get_feature_names())
    
    # Merge results into final dataframe
    result = pd.concat([csv_table, df], axis=1, sort=False)

    # Create the classification labels
    Y = result['class']

    IDs = result['ID']

    # Drop unnecessary fields
    result = result.drop('text', axis=1)
    result = result.drop('ID', axis=1)
    result = result.drop('class', axis=1)

    # Create a variable to hold the dataset
    X = result

    # Scale the dataset
    scaled = minmaxscale(result)

    # Load in the models
    mlp_opt = load("C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\Models\\mlp_opt.joblib")
    mlp = load("C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\Models\\mlp.joblib")
    rf = load("C:\\Users\\Kelly\\Desktop\\Programming Assignment 4\\Models\\rf.joblib")

    # Predict using the models
    mlp_opt_preds = mlp_opt.predict(scaled)
    mlp_preds = mlp.predict(scaled)
    rf_preds = rf.predict(scaled)

    # # Create precision, recall, and F1 metrics
    # mlp_opt_precision, mlp_opt_recall, mlp_opt_f1 = collect_metrics(Y, mlp_opt_preds)
    # mlp_precision, mlp_recall, mlp_f1 = collect_metrics(Y, mlp_preds)
    # rf_precision, rf_recall, rf_f1 = collect_metrics(Y, rf_preds)

    # # Pretty print the results
    # print("MLP OPT | Recall: {} | Precision: {} | F1: {}".format(mlp_opt_recall, mlp_opt_precision, mlp_opt_f1))
    # print("MLP     | Recall: {} | Precision: {} | F1: {}".format(mlp_recall, mlp_precision, mlp_f1))
    # print("RF      | Recall: {} | Precision: {} | F1: {}".format(rf_recall, rf_precision, rf_f1))

    # Print out the required dev test results
    mlp_opt_res = Y.to_frame('Actual Value')
    mlp_opt_res.insert(0, 'Predicted Value', mlp_opt_preds)
    mlp_opt_res = IDs.to_frame().join(mlp_opt_res, how='inner')
    print(mlp_opt_res)

    # # Print out the required dev test results
    # mlp_res = Y.to_frame('Actual Value')
    # mlp_res.insert(0, 'Predicted Value', mlp_preds)
    # mlp_res = IDs.to_frame().join(mlp_res, how='inner')
    # print(mlp_res)

    # # Print out the required dev test results
    # rf_res = Y.to_frame('Actual Value')
    # rf_res.insert(0, 'Predicted Value', rf_preds)
    # rf_res = IDs.to_frame().join(rf_res, how='inner')
    # print(rf_res)            

    # for name, pred in zip(IDs, mlp_opt_preds):
    #     print(name, "   ", pred)

main()
