# This script will create a Classification model to classify inbound messages into the appropriate category
# Uses the Natural Language Toolkit to provide the lexical analysis of the messages


# import libraries
# Need to sklearn libs for the model building
# Need Pickle lib to save model

import pandas as pd
import numpy as np
import sys
import os
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import time

# Using the Natural Language toolkit and "punkt" to chop up sentences and "wordnet" for the lexical corpus
# Great reference here: https://www.nltk.org/ and nifty tutorial here: https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/

import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    '''
    Load the database from the given filepath and process them as X, y and category_names
    Input: Database filepath
    Output: Returns the Features X (message) & target dataframe Y along with target columns names and category names
    '''
    
    print("About to read SQL Table")
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table("messages_disaster", engine)
    print("Did read SQL Table")
    #print("Df:", df)
    #print("Df Describe:", df.info())

    print("About to set X")
    X = df['message']
    print("X:", X)
    y = df.drop(["message","id","genre","original"], axis=1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    Function to tokenize the text messages
    Input: text
    output: cleaned tokenized text as a list object
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    Build the model

    Returns:
        pipeline (pipeline.Pipeline): model
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    #pipeline.get_params()
    # NOTE: I needed to use this shortened parameter list since the fuller parameter list ran too long and the session timed out
    parameters_short = {
    'clf__estimator__n_estimators': [10],
    'vect__ngram_range': [(1, 1)]
        }
    
    # Main parameters I would use with a longer work session
    parameters = {'vect__ngram_range': ((1, 1), (1, 2))
            , 'vect__max_df': (0.5, 0.75, 1.0)
            , 'tfidf__use_idf': (True, False)
            , 'clf__estimator__n_estimators': [50, 100, 200]
            , 'clf__estimator__min_samples_split': [2, 3, 4]
        }

    pipeline = GridSearchCV(pipeline, param_grid = parameters_short)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performances and print the results

        model: model to evaluate
        X_test: Test dataset
        Y_test: dataframe containing the categories
        category_names:  categories name
    '''
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
       print('Category: {} '.format(category_names[i]))
       print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
       print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))

def save_model(model, model_filepath):
    '''
    Function to save the model
    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
  
        start = time.process_time()
        print('Building model...')
        model = build_model()
        print("Build time",(time.process_time() - start))
        
        start = time.process_time()
        print('Training model...')
        model.fit(X_train, Y_train)
        print("Train time",(time.process_time() - start))
        
        start = time.process_time()
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print("Eval time",(time.process_time() - start))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()