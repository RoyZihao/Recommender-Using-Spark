#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Extension 1: Alternative Model Formulation

Usage:

    $ spark-submit extension_amf.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet 0.01 1 1 1

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


def main(spark, trainFilePath, valFilePath, downPercentage, dropn, perform_log, perform_drop):
    '''
    trainFilePath: path of training set
    
    valFilePath: path of validation file or test file, whichever the evaluation is based on.
    
    downPercentage: percentage of training set used to train the model. Use 1 if using the full set.
    
    dropn: number less or equal to this number will be dropped for the drop low count approach.
    
    perform_log: 1 if run the logarithm approach, 0 if not. 
    
    perform_drop: 1 if run the dropping approach, 0 if not. 
    
    outpuut: will print the precedure of the whole process, then print MAP and precision at 500 for the selected approach.
             Note: need to manually change the configuration wanted, at each config = [reg, rank, alpha].
    '''

    print('Alternative model formualtions...')

    # Get Train sample & Validation 
    dropn = int(dropn)
    perform_log = int(perform_log)
    perform_drop = int(perform_drop)
    downPercentage = float(downPercentage)
    if 'validation' in valFilePath:
        print('Using ' + str(downPercentage*100) + '% of the training set and the validation set...')
    else:
        print('Using ' + str(downPercentage*100) + '% of the training set and the test set...')
    
    
    
    # val_users_l = [row.user_index for row in val_users.collect()]

    # Initialize ALS Parameters # TODO: configutation on the parameters
    if perform_drop == 1:

        print('Eliminating count <= '+ str(dropn) + '...')
        print('Importing Data...')
        train = spark.read.parquet(trainFilePath)
        train_sample = train.sample(False, downPercentage, seed = 0)
        train= train.filter(train['count'] > dropn)
        val = spark.read.parquet(valFilePath)
        
        print('Building Indexers...')
        # Generate indexers and fit to train
        indexerUser = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='skip')
        indexerTrack = StringIndexer(inputCol="track_id", outputCol="track_index", handleInvalid='skip')

        indexers = Pipeline(stages = [indexerUser, indexerTrack])
        model = indexers.fit(train_sample)

        # Transform indexers to train sample and val
        train_sample = model.transform(train_sample)
        val = model.transform(val)

        # Get intersection of user indexs
        val_users = val.select('user_index').distinct()


        config = [0.01, 4, 0.01]
        print('Configuration: regParam = ' + str(config[0]) + ', rank = ' + str(config[1]) + ', alpha = ' + str(config[2]) + '.')
        print('Generating model...')
        als = ALS(alpha = config[2], rank = config[1], regParam=config[0], userCol="user_index", itemCol="track_index", ratingCol="count", coldStartStrategy="drop", implicitPrefs = True)
        model_als = als.fit(train_sample)

        print('Getting the Prediction list...')
        # Get top 500 recommended items for val users: Prediction List
        top500_val = model_als.recommendForUserSubset(val_users, 500).cache()
        predList = top500_val.select(top500_val.user_index,
        top500_val.recommendations.track_index.alias('pred_list'))
        print('Getting the True list...')

        # Build True List
        trueList = val.orderBy(col('user_index'))\
        .groupBy('user_index')\
        .agg(expr('collect_list(track_index) as true_list'))

        # Join The lists and generate RDD for ranking metric
        trueList = trueList.alias('trueList')
        predList = predList.alias('predList')
        predTrueList = predList.join(trueList, predList.user_index == trueList.user_index).select('predList.pred_list', 'trueList.true_list')
        predictionAndLabels = predTrueList.rdd.map(lambda row: (row.pred_list, row.true_list))

        print('Getting the evaluation...')
        # Build Evaluator and get MAP
        rankmetrics = RankingMetrics(predictionAndLabels)
        performance = [rankmetrics.meanAveragePrecision, rankmetrics.precisionAt(500)]
        print('The MAP is: ' + str(performance[0]))
        print('The Precision at 500 is: ' + str(performance[1]))


    if perform_log == 1: 
        print('Implementing Log-based methods')

        print('Importing Data...')
        train = spark.read.parquet(trainFilePath)
        train_sample = train.sample(False, downPercentage, seed = 0)
        train_sample = train_sample.withColumn("log_count", log('count'))
        train_sample = train_sample.withColumn("log_count_plus", log1p('count'))
        val = spark.read.parquet(valFilePath)
        
        print('Building Indexers...')
        # Generate indexers and fit to train
        indexerUser = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='skip')
        indexerTrack = StringIndexer(inputCol="track_id", outputCol="track_index", handleInvalid='skip')

        indexers = Pipeline(stages = [indexerUser, indexerTrack])
        model = indexers.fit(train_sample)

        # Transform indexers to train sample and val
        train_sample = model.transform(train_sample)
        val = model.transform(val)

        # Get intersection of user indexs
        val_users = val.select('user_index').distinct()


        print('Version One: Change count into log(count + 1)...')
        config = [0.01, 4, 0.01]
        print('Configuration: regParam = ' + str(config[0]) + ', rank = ' + str(config[1]) + ', alpha = ' + str(config[2]) + '.')
        print('Generating model...')
        als = ALS(alpha = config[2], rank = config[1], regParam=config[0], userCol="user_index", itemCol="track_index", ratingCol="log_count_plus", coldStartStrategy="drop", implicitPrefs = True)
        model_als = als.fit(train_sample)
        
        print('Getting the Prediction list...')
        # Get top 500 recommended items for val users: Prediction List
        top500_val = model_als.recommendForUserSubset(val_users, 500).cache()
        predList = top500_val.select(top500_val.user_index,
        top500_val.recommendations.track_index.alias('pred_list'))
        
        print('Getting the True list...')
        # Build True List
        trueList = val.orderBy(col('user_index'))\
        .groupBy('user_index')\
        .agg(expr('collect_list(track_index) as true_list'))
        # Join The lists and generate RDD for ranking metric
        trueList = trueList.alias('trueList')
        predList = predList.alias('predList')
        predTrueList = predList.join(trueList, predList.user_index == trueList.user_index).select('predList.pred_list', 'trueList.true_list')
        predictionAndLabels = predTrueList.rdd.map(lambda row: (row.pred_list, row.true_list))
        
        print('Getting the evaluation...')
        # Build Evaluator and get MAP
        rankmetrics = RankingMetrics(predictionAndLabels)
        performance = [rankmetrics.meanAveragePrecision, rankmetrics.precisionAt(500)]
        print('The MAP is: ' + str(performance[0]))
        print('The Precision at 500 is: ' + str(performance[1]))


        print('Version Two: Taking the Log of Count...')
        config = [0.01, 4, 0.01]
        print('Configuration: regParam = ' + str(config[0]) + ', rank = ' + str(config[1]) + ', alpha = ' + str(config[2]) + '.')
        print('Generating model...')
        als = ALS(alpha = config[2], rank = config[1], regParam=config[0], userCol="user_index", itemCol="track_index", ratingCol="log_count", coldStartStrategy="drop", implicitPrefs = True)
        model_als = als.fit(train_sample)
        
        print('Getting the Prediction list...')
        # Get top 500 recommended items for val users: Prediction List
        top500_val = model_als.recommendForUserSubset(val_users, 500).cache()
        predList = top500_val.select(top500_val.user_index,
        top500_val.recommendations.track_index.alias('pred_list'))
        
        # Join The lists and generate RDD for ranking metric
        predList = predList.alias('predList')
        predTrueList = predList.join(trueList, predList.user_index == trueList.user_index).select('predList.pred_list', 'trueList.true_list')
        predictionAndLabels = predTrueList.rdd.map(lambda row: (row.pred_list, row.true_list))
        
        print('Getting the evaluation...')
        # Build Evaluator and get MAP
        rankmetrics = RankingMetrics(predictionAndLabels)
        performance = [rankmetrics.meanAveragePrecision, rankmetrics.precisionAt(500)]
        print('The MAP is: ' + str(performance[0]))
        print('The Precision at 500 is: ' + str(performance[1]))



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('train_sample').getOrCreate()

    # Get the filename from the command line
    trainFilePath = sys.argv[1]

    # And the location to store the trained model
    valFilePath = sys.argv[2]

    downPercentage = sys.argv[3]

    dropn = sys.argv[4]

    perform_log = sys.argv[5]

    perform_drop = sys.argv[6]

    # Call our main routine
    main(spark, trainFilePath, valFilePath, downPercentage, dropn, perform_log, perform_drop)
