
import graph_maker
import pickle
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
import pyspark.ml.recommendation
from pyspark.sql.functions import isnan, col
from pyspark.ml.evaluation import RegressionEvaluator
import scipy.sparse as scs
import data_frame_creator
import sparse_matrix_functions

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import desc
#sc = SparkContext('local')
#spark = SparkSession(sc)
spark = pyspark.sql.SparkSession.builder.master("local[32]").getOrCreate()


with open("gene_dictionary_final.pickle", "rb") as dict_gene:
        gene_dict = pickle.load(dict_gene)

with open("inverse_network_genes.pickle", "rb") as inverse_dict_gene:
        inverse_network_genes = pickle.load(inverse_dict_gene)

with open("network_drugs.pickle", "rb") as drugs:
        network_drugs = pickle.load(drugs)


resist_network_matrix=data_frame_creator.open_pickle('spark/resist_network_matrix.pickle')
sensit_network_matrix=data_frame_creator.open_pickle('spark/sensit_network_matrix.pickle')
any_network_matrix=data_frame_creator.open_pickle('spark/any_network_matrix.pickle')

R = scs.coo_matrix(resist_network_matrix)
S = scs.coo_matrix(sensit_network_matrix)
A = scs.coo_matrix(any_network_matrix)

R_df=pd.DataFrame({'gene_id':R.row, 'drug_id':R.col, 'data':R.data})
R_df['log_data'] = np.log(R_df['data']+1)
spark_R_df = spark.createDataFrame(R_df)

S_df=pd.DataFrame({'gene_id':S.row, 'drug_id':S.col, 'data':S.data})
S_df['log_data'] = np.log(S_df['data']+1)
spark_S_df = spark.createDataFrame(S_df)

A_df=pd.DataFrame({'gene_id':A.row, 'drug_id':A.col, 'data':A.data})
A_df['log_data'] = np.log(A_df['data']+1)
spark_A_df = spark.createDataFrame(A_df)

ranks = [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
regs = [ 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

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


def make_model(X, rank, reg, path):
    filename=path+'.pickle'
    als_model = pyspark.ml.recommendation.ALS(
    itemCol='drug_id',
    userCol='gene_id',
    ratingCol='log_data',
    nonnegative=True,
    regParam=reg,
    rank=rank)
    drug_recommender = als_model.fit(X)
    #data_frame_creator.write_pickle(filename, drug_recommender)
    return drug_recommender





def predict_drugs(predict_gene, kind='r', drug_number=10, desc = False):
    gene =gene_dict[predict_gene]
    gen_num = inverse_network_genes[gene]
    if kind == 'r' or kind == 'R':
        col_num = resist_network_matrix.shape[1]
        model = pyspark.ml.recommendation.ALSModel.load('resist_NMF_model')
    if kind == 's' or kind == 'S':
        col_num = sensit_network_matrix.shape[1]
        model = pyspark.ml.recommendation.ALSModel.load('sensit_NMF_model')
    if kind == 'a' or kind == 'A':
        col_num = sensit_network_matrix.shape[1]
        model = pyspark.ml.recommendation.ALSModel.load('any_NMF_model')
    r = [gen_num for n in range(0,col_num)]
    c = [n for n in range(0,col_num)]
    test_df=pd.DataFrame({'gene_id':r,'drug_id': c})
    spark_test_df = spark.createDataFrame(test_df)
    prediction = model.transform(spark_test_df)
    selected = prediction.select("gene_id", "drug_id", "prediction")
    if desc:
        top = selected.sort(desc('prediction')).collect()[0:drug_number]
    else:
        top = selected.sort('prediction').collect()[0:drug_number]
    drug_nums = []
    for drug in top:
        print(drug)
        if not(np.isnan(drug[1])):
            drug_nums.append(drug[1])
    drug_lst = []
    for num in drug_nums:
        drug = network_drugs[num]
        drug_lst.append(drug[1:-1])
    print('the top {} predicted for gene {} are {}.'. format(drug_number, predict_gene, drug_lst))
    return drug_lst


def get_user_input():
    genes = input('For what genes would you like to get drug interaction information?:, enter "all" for full network  ')
    genes_list = graph_maker.process_genes(genes)
    kind = input('if you are interested on drug resistance evidence press "r"'
                'if you are interested on drug resistance evidence press "s"'
                 'if you are interested on general interactions press "g" '
                'if you are interested on all of above interactions press "a" ')
    drug_number = input('How many drug names would you like to see? ' )
    order = input ('Would you like to display the drugs in ascending or descendicg order. a/d: ')
    if order == 'd' or order == 'D':
        desc = True
    else:
        desc = False
    return genes_list, kind, drug_number, desc



def provide_drug_predictions():
    genes_list, kind, drug_number, desc = get_user_input():
    predict_dict={}
    for gene in gene_list:
        drug_lst = predict_drugs(gene, kind, drug_number, desc):
        predict_dict[gene] = drug_lst
    return predict_dict
