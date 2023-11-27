from pyspark.sql import SparkSession

INDEX_NAME = 'spark_index'

spark = SparkSession.Builder() \
    .appName("Binh_test") \
    .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.6.2") \
    .master("local[4]").getOrCreate()

df = spark.read.format('csv').option('header', 'true').load(
    'data/training_pool.csv')
    
df = df.select([col for col in df.columns if col != 'target'])

df.write.format("org.elasticsearch.spark.sql") \
    .option("es.resource", '%s/%s' % (INDEX_NAME, "doc")) \
    .mode("overwrite") \
    .save()

print('done')
