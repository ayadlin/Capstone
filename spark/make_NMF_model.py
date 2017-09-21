
#from pyspark import SparkContext, SparkConf
#sc =SparkContext()
import pickle
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
import pyspark.ml.recommendation
from pyspark.sql.functions import isnan, col
from pyspark.ml.evaluation import RegressionEvaluator
import scipy.sparse as scs
#import data_frame_creator
#import sparse_matrix_functions

#resist_network_matrix=data_frame_creator.open_pickle('resist_network_matrix.pickle')
#sensit_network_matrix=data_frame_creator.open_pickle('sensit_network_matrix.pickle')
#any_network_matrix=data_frame_creator.open_pickle('any_network_matrix.pickle')

with open('resist_network_matrix.pickle', 'rb') as handle:
    resist_network_matrix=pickle.load(handle)

with open('sensit_network_matrix.pickle', 'rb') as handle:
    sensit_network_matrix=pickle.load(handle)

with open('any_network_matrix.pickle', 'rb') as handle:
    any_network_matrix=pickle.load(handle)

R = scs.coo_matrix(resist_network_matrix)
S = scs.coo_matrix(sensit_network_matrix)
A = scs.coo_matrix(any_network_matrix)

R_df=pd.DataFrame({'gene_id':R.row, 'drug_id':R.col, 'data':R.data})
R_df['log_data'] = np.log(R_df['data']+1)
spark_R_df = spark.createDataFrame(R_df)

S_df=pd.DataFrame({'gene_id':S.row, 'drug_id':S.col, 'data':S.data})
S_df['log_data'] = np.log(S_df['data']+1)
spark_S_df = spark.createDataFrame(S_df)

A_df=pd.DataFrame({'gene_id':A.row, 'drug_id':A.col, 'data':R.data})
A_df['log_data'] = np.log(A_df['data']+1)
spark_A_df = spark.createDataFrame(A_df)

ranks = [5,10,15,20,25,30,50]
regs = [0, 0.001, 0.01, 0.1]

def best_model_values (spark_df, ranks, regs):
    train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=427471138)
    model_rmse=1000000
    params = []
    for rank in ranks:
        for reg in regs:
            als_model = pyspark.ml.recommendation.ALS(
            itemCol='drug_id',
            userCol='gene_id',
            ratingCol='log_data',
            nonnegative=True,
            regParam=reg,
            rank=rank)
            drug_recommender = als_model.fit(train_df)
            drug_predictions = drug_recommender.transform(test_df)
            drug_pred_not_null = drug_predictions.filter(~isnan(drug_predictions["prediction"]))
            #drug_pred_not_null.show()
            rmse_eval=RegressionEvaluator(metricName="rmse",  labelCol='log_data', predictionCol='prediction')
            error = rmse_eval.evaluate(drug_pred_not_null)
            if error < model_rmse:
                #model = als_model.fit(spark_df)
                model_rank = rank
                model_regParam = reg
                model_rmse = error
            params.append([rank, reg, error])
    return [(model_rank, model_regParam, model_rmse), params] #model,


def best_models(R, S, A, ranks, regs):
    best_R = best_model_values (R, ranks, regs)
    best_S = best_model_values (S, ranks, regs)
    best_A = best_model_values (A, ranks, regs)
    return [bestR, best_S, best_A]
