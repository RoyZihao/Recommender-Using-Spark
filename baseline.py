#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Recommender Model Training and Hyperparameter Tuning

Usage:

    $ spark-submit baseline.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet 0.01

'''

# We need sys to get the command line arguments
import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics
import numpy as np
import itertools
import operator



def main(spark, trainFilePath, valFilePath, downPercentage):
    '''
    trainFilePath: path of training set
    
    valFilePath: path of validation file or test file, whichever the evaluation is based on.
    
    downPercentage: percentage of training set used to train the model. Use 1 if using the full set.
    
    output: Print the precedure of the whole process, including MAP and precision at 500 for each configuration. At last, print the dictionary containing all configuration and scores, and then print the configuration that has the highest MAP score and the score itself.
    '''

    # Get Train sample & Validation 
    downPercentage = float(downPercentage)
    if 'validation' in valFilePath:
        print('Using ' + str(downPercentage*100) + '% of the training set and the validation set...')
    else:
        print('Using ' + str(downPercentage*100) + '% of the training set and the test set...')
    train = spark.read.parquet(trainFilePath)
    train_sample = train.sample(False, downPercentage, seed = 0)
    val = spark.read.parquet(valFilePath)

    # Generate indexers and fit to train
    indexerUser = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='skip')
    indexerTrack = StringIndexer(inputCol="track_id", outputCol="track_index", handleInvalid='skip')

    print('Generate the model for transforming user_id and track_id')
    indexers = Pipeline(stages = [indexerUser, indexerTrack])
    model = indexers.fit(train_sample)

    # Transform indexers to train sample and val
    print('Transform user_id and track_id into numerical values')
    train_sample = model.transform(train_sample)
    val_sample = model.transform(val)

    # Get intersection of user indexs
    valUsers = val_sample.select('user_index').distinct()

    # Initialize ALS Parameters 
    param = [[0.001,0.01,0.1],[5,8,10],[0.3,0.7,1]]
    config = list(itertools.product(*param))

    print('Hyper-parameter tuning...')
    performance = {}
    # Grid Search
    for conf in config:
        print('Configuration: regParam = ' + str(conf[0]) + ', rank = ' + str(conf[1]) + ', alpha = ' + str(conf[2]) + '.')
        print('Generating model...')
        als = ALS(alpha = conf[2], rank = conf[1], regParam=conf[0], userCol="user_index", itemCol="track_index", ratingCol="count", coldStartStrategy="drop", implicitPrefs = True)
        model_als = als.fit(train_sample)
        
        print('Getting the Prediction list...')
        # Get top 500 recommended items for val users: Prediction List
        top500_val = model_als.recommendForUserSubset(valUsers, 500).cache()
        predList = top500_val.select(top500_val.user_index,
        top500_val.recommendations.track_index.alias('pred_list'))
        
        print('Getting the True list...')
        # Build True List
        trueList = val_sample.groupBy('user_index')\
        .agg(expr('collect_list(track_index) as true_list'))
        # Join The lists and generate RDD for ranking metric
        trueList = trueList.alias('trueList')
        predList = predList.alias('predList')
        predTrueList = predList.join(trueList, predList.user_index == trueList.user_index).select('predList.pred_list', 'trueList.true_list')
        predictionAndLabels = predTrueList.rdd.map(lambda row: (row.pred_list, row.true_list))
        
        print('Getting the evaluation...')
        # Build Evaluator and get MAP
        rankmetrics = RankingMetrics(predictionAndLabels)
        performance[conf] = [rankmetrics.meanAveragePrecision, rankmetrics.precisionAt(500)]
        print('The MAP is: ' + str(performance[conf][0]))
        print('The Precision at 500 is: ' + str(performance[conf][1]))

    print(performance)
    best_config = list(performance.keys())[np.argmax([i[0] for i in performance.values()])]
    print('The best MAP performance comes from the configuration: regParam = ' + str(best_config[0]) + ', rank = ' + str(best_config[1]) + ', alpha = ' + str(best_config[2]) + '.')
    print('The MAP is: ' + str(performance[best_config][0]) + '.')





# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('train_sample').getOrCreate()

    # Get the filename from the command line
    trainFilePath = sys.argv[1]

    # And the location to store the trained model
    valFilePath = sys.argv[2]

    downPercentage = sys.argv[3]

    # Call our main routine
    main(spark, trainFilePath, valFilePath, downPercentage)


