import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Docstrings: Adding these in now (based on last project, suggestion was use them almost exclusively for commenting/usage info)
    input:
        messages_filepath: The path of messages dataset.
        categories_filepath: The path of categories dataset.
    output:
        df: The merged dataset
    '''
    #load data from csv
    messages = pd.read_csv(messages_filepath) #load messages data from csv
    categories = pd.read_csv(categories_filepath) #load categories data from csv
    #df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    df = messages.merge(categories, on=["id"])
    return df


def clean_data(df):
    '''
    input: Dataframe to clean    
    output: Cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # Get first row of the categories dataframe
    Cat_Split_row = categories.head(1)

    # Slice Cat_Split_row taking off the last 2 chars

    category_colnames = Cat_Split_row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()

    # Now use this sliced row to rename columns of `categories`
    categories.columns = category_colnames
    
    #Fix the categories columns name
    for column in categories:
        # get and set using last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # change column to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    # concatenate the original dataframe with the new `categories` dataframe
    
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    input: Dataframe to save and the filename of the database that will be created (from the input arguments)    
    output: Cleaned dataframe
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    # Note I "replace" the table if it already exists
    df.to_sql('messages_disaster', engine, index=False,if_exists='replace')


def main():
    '''
    Arguments:  messages and category datasets as the first and second argument respectively
                filepath of the database to save the cleaned data 
    To run this script: $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()