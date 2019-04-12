# import libraries
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pyspark.sql.functions as psqf
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.mllib.tree import RandomForest, RandomForestModel, LabeledPoint

from pyspark.mllib.evaluation import BinaryClassificationMetrics

import datetime


def get_label_df(df):
    """
    Given sparkify data label each user as churned/ not churned.
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        churn_event: DataFrame
    """
    churn_event = df.groupby('userId').agg(psqf.collect_list('page').alias('pages'))
    # define 1 as churned, 0 otherwise
    churn_f = psqf.udf(lambda x: 1 if 'Cancel' in set(x) else 0)
    churn_event = churn_event.withColumn("label", churn_f(churn_event.pages)).drop('pages')
    return churn_event


def get_songs_played(df):
    """
    Given sparkify data count songs each user played.
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        songsplayed: DataFrame
    """
    songsplayed = df.where(psqf.col('song')!='null').groupby("userId").agg(psqf.count(psqf.col('song')).alias('SongsPlayed')).orderBy('userId')
    return songsplayed


def get_hour_counts(df):
    """
    Given sparkify data count unique hour of day each user has been active.
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        hour_count_df: DataFrame
    """

    hours_udf = psqf.udf(
        lambda x: datetime.datetime.utcfromtimestamp(x / 1000.0).strftime(
            '%Y-%m-%d-%H'))
    hours_df = df.select('userId', 'ts').withColumn('hour',
                                                    hours_udf(psqf.col('ts')))
    hour_count_df = hours_df.where(psqf.col('userId') != 'null').groupby(
        'userId').agg(
        (psqf.countDistinct(psqf.col('hour'))).alias("HourCount")).orderBy(
        'userId')
    return hour_count_df


def get_thubms_up_counts(df):
    """
    Given sparkify data count number of Thumbs up by each user.
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        thumbsup_count: DataFrame
    """
    thumbsup_count = df.where(
        (psqf.col('page') == 'Thumbs Up') & (psqf.col('userId') != 'null')) \
        .groupby("userId").agg(
        psqf.count(psqf.col('page')).alias('thumbsUpCount')).orderBy('userId')
    return thumbsup_count


def get_thumbs_down_counts(df):
    """
    Given sparkify data count number of Thumbs down by each user.
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        thumbsdown_count: DataFrame
    """
    thumbsdown_count = df.where(
        (psqf.col("page") == 'Thumbs Down') & (psqf.col('userId') != 'null')) \
        .groupby("userId").agg(
        psqf.count(psqf.col('page')).alias('thumbsDownCount')).orderBy('userId')
    return thumbsdown_count


def data_pipeline(churn_event, songsplayed, hour_count_df, thumbsup_count, thumbsdown_count):
    """
    Given  churn_event, songsplayed, hour_count_df, thumbsup_count,
    thumbsdown_count dataframes, join them and normalize the resulting
    DataFrame.
    Parameters
    -----------
        churn_event: DataFrame
            labeled users
        songsplayed: DataFrame
            songs count played by each user
        hour_count_df: DataFrame
            hour of day count a user has been active
        thumbsup_count: DataFrame
            Thummbs up count by each user
        thumbsdown_count: Dataframe
            Thumbs down count by each user
    returns
    -------
        input_data: DataFrame
            scaled DataFrame
    """
    features_df = churn_event.join(songsplayed, "userId")\
    .join(hour_count_df, "userId").join(thumbsup_count, "userId")\
    .join(thumbsdown_count, "userId")
    assembler = VectorAssembler(inputCols=["SongsPlayed", "HourCount", "thumbsUpCount", "thumbsDownCount"], outputCol="rawFeatures")
    features_df = assembler.transform(features_df)
    scaler = StandardScaler(inputCol="rawFeatures", outputCol="features", withStd=True)
    scalerModel = scaler.fit(features_df)
    features_df = scalerModel.transform(features_df)
    input_data = features_df.select('features', 'label')
    return input_data


def split_data(input_data):
    """
    Given input dataframe, convert to rdd, lablepoint and split into
    training, validation and test dataframes
    Parameters
    -----------
        input_data: DataFrame
    returns
    -------
        trainingData: DataFrame
        validationData: DataFrame
        testData: DataFrame
    """
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(input_data)
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(input_data)
    # Split the data into training and test sets (since dataset is imbalanced)
    (training_data, temp_data) = input_data.randomSplit([0.6, 0.4])
    (validation_data, test_data) = temp_data.randomSplit([0.5, 0.5])
    return training_data, validation_data, test_data, labelIndexer, featureIndexer


def train_model(training_data,labelIndexer,featureIndexer, model):
    """
    Train a RandomForest model using training data and score on validation data.
    Parameters
    -----------
        training_data: DataFrame
        validation_data: DataFrame
    returns
    -------
        None
    """
    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)
    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
def evaluate_model(model, data):
    """
    Make prediction and evaluate model.
    Parameters
    -----------
        model: model object
    returns
    -------
        None
    """
    predictions = model.transform(data)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Validation Error = %g" % (1.0 - accuracy))
    f1_score_evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",metricName='f1')
    f1_score = f1_score_evaluator.evaluate(predictions)
    print("F1 score = %g" % (f1_score))
    
    
if __name__=='__main__':
    # create a Spark session
    spark = SparkSession.builder.appName(
        "customer-churn data pipeline").getOrCreate()
    data_path = "mini_sparkify_event_data.json"
    df = spark.read.json(data_path)
    churn_df = get_label_df(df)
    songs_df = get_songs_played(df)
    hours_df = get_hour_counts(df)
    up_df = get_thubms_up_counts(df)
    down_df = get_thumbs_down_counts(df)
    input_data = data_pipeline(churn_df, songs_df, hours_df, up_df, down_df)
    
    train_data, valid_data, test_data, labelIndexer, featureIndexer = split_data(input_data)
    rfc = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rfc, labelConverter])
    param_grid = ParamGridBuilder().addGrid(rfc.numTrees, [10, 15]).addGrid(rfc.maxDepth, [2, 5]).build()
    cv = CrossValidator(estimator=pipeline, 
                        estimatorParamMaps = param_grid, 
                        evaluator = MulticlassClassificationEvaluator(metricName='f1'),
                        numFolds=3)

    best_model = cv.fit(trainingData)
    evaluate_model(model=best_model, data=test_data)

    