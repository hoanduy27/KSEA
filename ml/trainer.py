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

import math

import pandas as pd
from modAL.models import ActiveLearner
from ml.preprocess import feature_extraction


tolist_udf = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))

# def extract_feature(pipeline_model, df):
#     pipeline_model.transform(df).withColumn("features", tolist_udf("features")).cache()

@pandas_udf("boolean")
def to_query(features_series, self):
    X_i = np.stack(features_series.to_numpy())
    n = X_i.shape[0]
    query_idx, _ = self.learner.query(X_i, n_instances=math.ceil(n * self.query_factor))
    # Output has same size of inputs; most instances were not sampled for query
    query_result = pd.Series([False] * n)
    # Set True where ActiveLearner wants a label
    query_result.iloc[query_idx] = True
    return query_result

class LabelSimulator:
    def __init__(self, label_data_path):
        self.label_df = pd.read_csv(label_data_path)

    def annotate(self, ids):
        return self.label_df[self.label_df.isin(ids)].target


class Strategy:
    def __init__(
            self, 
            pre_model, 
            labeled_df: DataFrame, 
            unlabeled_df: DataFrame, 
            labeler: LabelSimulator, 
            query_factor=0.1
        ):
        self.pre_model = pre_model
        self.labeled_df = labeled_df 
        self.unlabeled_df = unlabeled_df
        self.learner = ActiveLearner(
            estimator=pre_model,
        )
        self.query_factor = query_factor

    def query(self):

        # tolist_udf = F.udf(
        #     lambda v: v.toArray().tolist(), 
        #     ArrayType(FloatType())
        # )
        unlabeled_df_tr = feature_extraction(self.unlabeled_df)\
                            .withColumn("features", tolist_udf("features")) \
                            .select("*") \
                            .withColumn("query", to_query("features", self)).cache() \
                            .filter("query").select("text")
        
        print(unlabeled_df_tr.show())

class Trainer:
    def __init__(self, df, val_size, test_size):
        self.df = df 
        self.val_size = val_size
        self.test_size = test_size
        
        (
            self.X_train, self.X_val, self.X_test, 
            self.y_train, self.y_val, self.y_test, 
        ) = self._split_dataset()

        self.hparams = None 
        self.model = None 

    def _split_dataset(self):
        X = np.stack(self.df["features"].to_numpy())
        y = self.df["target"].to_numpy()

        (X_train, X2, y_train, y2) = train_test_split(
            X, y, 
            test_size= self.val_size + self.test_size, 
            stratify=y, 
            random_state=7
        )

        (X_val, X_test, y_val, y_test) = train_test_split(
            X2,
            y2,
            test_size=self.test_size / (self.val_size + self.test_size),
            stratify=y2,
            random_state=7,
        )
        return (X_train, X_val, X_test, y_train, y_val, y_test)
    
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




def log_and_eval_model(best_model, best_params, X_test, y_test):
    with mlflow.start_run():
        accuracy = accuracy_score(y_test, best_model.predict(X_test))
        f1 = f1_score(y_test, best_model.predict(X_test), pos_label=4)
        loss = log_loss(y_test, best_model.predict_proba(X_test))
        mlflow.log_params(best_params)
        mlflow.log_metrics({"accuracy": accuracy, "log_loss": loss})
        mlflow.sklearn.log_model(best_model, "model")
        return (accuracy, f1, loss)

if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.Builder() \
        .appName("data_processor") \
        .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.6.2") \
        .master("local[4]").getOrCreate()

    # df = spark.read.format('csv').option('header', 'true').load(
    #     'data/complaints_init.csv'
    # )

    df = spark.read.csv(
        'data/training_init.csv',
        header=True,
        inferSchema=True    
    )

    df_unlab = spark.read.csv(
        'data/training_sorted.csv',
        header=True,
        inferSchema=True
    )

    df_unlab.sample(0.01)

    df_tr = feature_extraction(df).withColumn("features", tolist_udf("features")).cache()

    trainer = Trainer(
        df_tr.toPandas(), 0.1, 0.1
    )

    trainer.find_best_lr_model()

    print(log_and_eval_model(trainer.model, trainer.hparams, trainer.X_test, trainer.y_test))

    strategy = Strategy(
        trainer.model,
        df,
        df_unlab,
        None
    )

    strategy.query()