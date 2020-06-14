# Disaster Response Pipeline Project

![Floating Bin](https://media.giphy.com/media/853jNve3ljqrYrcSOK/giphy.gif)

**Project Overview**

My Github Link: https://github.com/nameisunique/Disaster_Response_Pipeline

This project will analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

We were provided a data set containing real messages that were sent during disaster events. Using this data set as well as a dataset describing categories, we create a machine learning pipeline to categorize these events so that messages are sent to an appropriate disaster relief agency.

We also use a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

If you are running on a Udacity terminal you will need to do the following:

1) Run your app with python run.py command
2) Open another terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
3) Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id that you got in the step 2
4) Press enter and the app should now run for you

