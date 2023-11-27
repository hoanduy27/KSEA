# init model
from typing import Optional
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer
from pyspark.ml.param.shared import (
    HasInputCol,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from pyspark.sql.types import StringType, ArrayType, FloatType

import pyspark.sql.functions as F

import re
import string

from pyspark.sql.dataframe import DataFrame
from ml.slang_words import slang, emojis


class Preprocessor(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
):
    input_col = Param(
        Params._dummy(),
        "input_col",
        "input column name.",
        typeConverter=TypeConverters.toString,
    )
    output_col = Param(
        Params._dummy(),
        "output_col",
        "output column name.",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(self, input_col: str = "input", output_col: str = "output"):
        super(Preprocessor, self).__init__()
        self._setDefault(input_col=None, output_col=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, input_col: str = "input", output_col: str = "output"):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def get_input_col(self):
        return self.getOrDefault(self.input_col)

    def get_output_col(self):
        return self.getOrDefault(self.output_col)

    def remove_html_tags(self, text):
        pattern = re.compile("<.*?>")
        return pattern.sub(r"", text)

    def remove_url(self, text):
        pattern = re.compile(r"https?://\S+|www\.\S+")
        return pattern.sub(r"", text)

    def handle_emoji(self, text):
        for emoji in emojis.keys():
            text = text.replace(emoji, "EMOJI" + emojis[emoji])

        return text

    def chat_conversion(self, text):
        new_text = []
        for w in text.split():
            if w.upper() in slang:
                new_text.append(slang[w.upper()])
            else:
                new_text.append(w)
        return " ".join(new_text)

    def remove_punc(self, text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def lowercase(self, text):
        return text.lower()

    def preprocess(self, text):
        text = self.remove_html_tags(text)
        text = self.remove_url(text)
        text = self.handle_emoji(text)
        text = self.chat_conversion(text)
        text = self.remove_punc(text)
        text = self.lowercase(text)

        return text

    def _transform(self, df: DataFrame) -> DataFrame:
        input_col = self.get_input_col()
        output_col = self.get_output_col()

        transform_udf = F.udf(self.preprocess, StringType())

        return df.withColumn(output_col, transform_udf(input_col))


def feature_extraction(df):
    preprocessor = Preprocessor(input_col="text", output_col="text_clean")
    # Tokenize into words
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="tokenized")
    # Remove stopwords
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
    # Compute term frequencies and hash into buckets
    hashing_tf = HashingTF(
        inputCol=tokenizer.getOutputCol(), outputCol="hashed", numFeatures=1000
    )
    # Convert to TF-IDF
    idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="features")

    pipeline = Pipeline(stages=[preprocessor, tokenizer, remover, hashing_tf, idf])
    pipeline_model = pipeline.fit(df)


    return pipeline_model.transform(df)

