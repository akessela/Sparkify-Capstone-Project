# Sparkify-Capstone-Project
Determine customer churn for a music streaming company called Sparkify.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
This project uses the following software and python libraries

- [Python 3.6](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [Pyspark-2.4.1](https://spark.apache.org/docs/latest/api/python/index.html)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)
  

## Project Motivation<a name="motivation"></a>
Imagine you are working for music streaming company like Spotify or Pandora called Sparkify.Millions of users stream thier favorite songs everyday. Each user  uses either the Free-tier with advertisement between the songs or the premium Subscription Plan.Users can upgrade, downgrade or cancel thier service at any time.Hence, it's crucial to make sure that users love the service provided by Sparkify. Every time a users interacts with the Sparkify app data is generated. Events such as playing a song, logging out, like a song etc. are all recorded. All this data contains key insights that can help the business thrive. The goal of this project is then to analyse this data and predict which group of users are expected to churn - either downgrading from premium to free or cancel thier subscriptions altogether.

In this project we'll be performing the following tasks:
1. Data Exploration
    - Learn about the data
2. Define Churn and label data based on churn definition
    - Determine which feature or feature value can be user to defin churn
3. Feature Engineering
    - Create features for each user. This data will be used as input to the model
4. Data transformation, data splitting and model training
    - Transform feature engineered data. 
    - Split data into training, validation and test data.
    - Build a machine learning model to train using training data

## File Descriptions <a name="files"></a>
* Sparkify-Capstone-Project : Jputer notebook used for this project
* main_sparkify.py: python script extracted from the notebook

## Results<a name="results"></a>

Please see notebook for analysis and results.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to [UC Irvine](https://archive.ics.uci.edu/ml/datasets/Census+Income) for the data and   [Udacity](https://www.udacity.com/courses/all) for creating a beautiful learning experience.  Find the Licensing for the data and other descriptive information from [UC Irvine ML Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income).
