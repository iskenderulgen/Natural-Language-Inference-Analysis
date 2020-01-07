from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

path = "/home/ulgen/Documents/Python_Projects/Contradiction/data/"

conf = SparkConf().setMaster("local[*]") \
    .setAppName("Contradiction Pre Process") \
    .set("spark.rdd.compress", "true") \
    .set("spark.driver.memory", "40g")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .getOrCreate()

contradiction = spark.read.json(path + "dev/contradiction.json").withColumnRenamed('similarity',
                                                                                   'contradiction_similarity')
entailment = spark.read.json(path + "dev/entailment.json").withColumnRenamed('similarity',
                                                                             'entailment_similarity')
neutral = spark.read.json(path + "dev/neutral.json").withColumnRenamed('similarity', 'neutral_similarity')

contradiction.show(5)
entailment.show(5)
neutral.show(5)

contra_sim = contradiction.select("contradiction_similarity").toPandas().to_numpy()
entail_sim = entailment.select("entailment_similarity").toPandas().to_numpy()
neutral_sim = neutral.select("neutral_similarity").toPandas().to_numpy()


fig1, ax1 = plt.subplots()
ax1.set_title('contradiction_similarity')
ax1.boxplot(contra_sim)

fig2, ax1 = plt.subplots()
ax1.set_title('entailment_similarity')
ax1.boxplot(entail_sim)

fig3, ax1 = plt.subplots()
ax1.set_title('neutral_similarity')
ax1.boxplot(neutral_sim)

plt.show()
