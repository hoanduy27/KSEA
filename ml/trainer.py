
import math
from math import sqrt
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, f1_score
from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK

from pyspark.sql.functions import pandas_udf
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType, ArrayType, FloatType
import pyspark.sql.functions as F

import pandas as pd
from modAL.models import ActiveLearner
from ml.preprocess import feature_extraction

from pyspark.sql import SparkSession


spark = SparkSession.Builder() \
    .appName("data_processor") \
    .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.6.2") \
    .master("local[4]").getOrCreate()


tolist_udf = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))

# def extract_feature(pipeline_model, df):
#     pipeline_model.transform(df).withColumn("features", tolist_udf("features")).cache()
# def to_query(strategy):

class LabelSimulator:
    def __init__(self, label_data_path):
        # self.label_df = pd.read_csv(label_data_path)
        self.label_df = spark.read.csv(
            label_data_path,
            header=True,
            inferSchema=True
        )

    def annotate(self, unlabeled_df):
        # return self.label_df.where(self.label_df['id'].isin(ids)).target

        return self.label_df\
                .join(unlabeled_df, on=["id", "user"])\
                .dropDuplicates() \
                .select("target", "id", "date", "flag", "user", "text")
        # return self.label_df[self.label_df.isin(ids)].target
    
class Trainer:
    def __init__(self, df, val_size):
        self.df = df 
        self.val_size = val_size
        
        (
            self.X_train, self.X_val, 
            self.y_train, self.y_val, 
        ) = self._split_dataset()

        self.hparams = None 
        self.model = None 

    def _split_dataset(self):
        X = np.stack(self.df["features"].to_numpy())
        y = self.df["target"].to_numpy()

        (X_train, X_val, y_train, y_val) = train_test_split(
            X, y, 
            test_size= self.val_size, 
            stratify=y, 
            random_state=7
        )

        return (X_train, X_val, y_train, y_val)
    
    # Core function to train a model given train set and params
    def train_model(self, params, X_train, y_train):
        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            # penalty=params["penalty"],
            penalty="l2",
            C=params["C"],
            random_state=7,
        )
        return lr.fit(X_train, y_train)


    # Use hyperopt to select a best model, given train/validation sets
    def find_best_lr_model(self):
        # Wraps core modeling function to evaluate and return results for hyperopt
        def train_model_fmin(params):
            lr = self.train_model(params, self.X_train, self.y_train)
            loss = log_loss(self.y_val, lr.predict_proba(self.X_val))
            accuracy = accuracy_score(self.y_val, lr.predict(self.X_val))
            f1 = f1_score(self.y_val, lr.predict(self.X_val), pos_label=4)
            # supplement auto logging in mlflow with f1 and accuracy
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("accuracy", accuracy)
            return {"status": STATUS_OK, "loss": loss, "accuracy": accuracy, "f1": f1}

        # penalties = ["l1", "l2", "elasticnet"]
        search_space = {
            "C": hp.loguniform("C", -6, 1),
            # "penalty": hp.choice("penalty", penalties),
        }

        best_params = fmin(
            fn=train_model_fmin,
            space=search_space,
            algo=tpe.suggest,
            max_evals=1,
            trials=SparkTrials(parallelism=4),
            rstate=np.random.default_rng(7),
        )
        # Need to translate this back from 0/1 in output to be used again as input
        # best_params["penalty"] = penalties[best_params["penalty"]]
        # Train final model on train + validation sets
        final_model = self.train_model(
            best_params, np.concatenate([self.X_train, self.X_val]), np.concatenate([self.y_train, self.y_val])
        )
        self.hparams = best_params
        self.model = final_model


class Strategy:
    def __init__(
            self, 
            labeled_df: DataFrame, 
            unlabeled_df: DataFrame, 
            labeler: LabelSimulator, 
        ):
        # self.pre_model = pre_model
        self.labeled_df = labeled_df 
        self.unlabeled_df = unlabeled_df
        
        self.labeler = labeler


    def to_query(self, features_series):
        @pandas_udf("boolean")
        def to_query_(features_series):
            X_i = np.stack(features_series.to_numpy())
            n = X_i.shape[0]
            
            query_idx, _ = learner.query(X_i, n_instances=math.ceil(n * QUERY_FACTOR))
            # Output has same size of inputs; most instances were not sampled for query
            query_result = pd.Series([False] * n)
            # Set True where ActiveLearner wants a label
            query_result.iloc[query_idx] = True
            return query_result
        
        return to_query_(features_series)



    def query(self):

        # tolist_udf = F.udf(
        #     lambda v: v.toArray().tolist(), 
        #     ArrayType(FloatType())
        # )
        unlabeled_df_tr = feature_extraction(self.unlabeled_df)\
                            .withColumn("features", tolist_udf("features")) \
                            .select("*") \
                            .withColumn("query", self.to_query("features")).cache() \
                            .filter("query") \
                            .select("id", "date", "flag", "user", "text")
        
        print(unlabeled_df_tr.count())
        unlabeled_df_tr
        
        # print("CHUA TRAN RAM")
        
        new_label_df = self.labeler.annotate(
            unlabeled_df_tr
        )
        
        new_label_df

        return new_label_df.union(df)

        # return new_label_df


def log_and_eval_model(best_model, best_params, X_test, y_test):
    with mlflow.start_run():
        accuracy = accuracy_score(y_test, best_model.predict(X_test))
        f1 = f1_score(y_test, best_model.predict(X_test), pos_label=4)
        loss = log_loss(y_test, best_model.predict_proba(X_test))
        mlflow.log_params(best_params)
        mlflow.log_metrics({"accuracy": accuracy, "log_loss": loss})
        mlflow.sklearn.log_model(best_model, "model")

        return (accuracy, f1, loss)

QUERY_FACTOR = 0.1
TEST_INDEX = 'test-kakfa-ingestion-flow/_doc'
POOL_INDEX = 'spark_index/doc'

df = spark.read.csv(
        'data/training_init.csv',
        header=True,
        inferSchema=True    
)

df_unlab = spark.read.format("org.elasticsearch.spark.sql") \
                .option("es_resource", POOL_INDEX) \
                .load().drop("target")

df_test = spark.read.format("org.elasticsearch.spark.sql") \
                .option("es_resource", TEST_INDEX).load()

# train init model

## feat extract
df_tr = feature_extraction(df)\
            .withColumn("features", tolist_udf("features")).cache()
df_test_tr = feature_extraction(df_test)\
                .withColumn("features", tolist_udf("features")).cache()\
                .toPandas()

X_test = np.stack(df_test_tr["features"].to_numpy())
y_test = df_test_tr["target"].to_numpy()

## start training
trainer = Trainer(
    df_tr.toPandas(), val_size=0.1
)
trainer.find_best_lr_model()

learner = ActiveLearner(
    estimator=trainer.model,
)

print(log_and_eval_model(learner.estimator, trainer.hparams, X_test, y_test))

# Active learning
strategy = Strategy(
    df,
    df_unlab,
    LabelSimulator('data/label.csv')
)

new_df = strategy.query()

## 
new_df_tr = feature_extraction(new_df).withColumn("features", tolist_udf("features")).cache()


## start training
trainer = Trainer(
    new_df_tr.toPandas(), val_size=0.1
)
trainer.find_best_lr_model()

learner = ActiveLearner(
    estimator=trainer.model,
)

print(log_and_eval_model(learner.estimator, trainer.hparams, X_test, y_test))
