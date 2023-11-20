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





