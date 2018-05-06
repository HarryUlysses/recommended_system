# coding=utf-8
from pyspark import SparkContext, sql

# from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.sql import Row
from pyspark.sql import SQLContext

sc = SparkContext('local', 'test')
sqlContext = sql.SQLContext(sc)
# >>> list = ["Hadoop","Spark","Hive","Spark"]
# >>> rdd = sc.parallelize(list)


rawData = sc.textFile("../DataSets/MovieLens/ml-20m/ratings.csv")
header = rawData.first()
rawData = rawData.filter(lambda row: row != header)
rawRatings = rawData.map(lambda x: x.split(','))
# rawRatings.foreach(lambda x: print x)

# print(rawRatings.take(5))
#
# ratings = rawRatings.map(lambda x: Row(user=int(x[0]), item=int(x[1]), rating=float(x[2])))
#
# ratings_df = sqlContext.createDataFrame(ratings)
# # ratings_df.show()
#
# training, test = ratings_df.randomSplit([0.8, 0.2])
# # training.show()
# # # print(test.take(5))
# alsExplicit = ALS(maxIter=5, regParam=0.01, userCol="user", itemCol="item", ratingCol="rating")
# alsImplicit = ALS(maxIter=5, regParam=0.01, implicitPrefs=True, userCol="user", itemCol="item", ratingCol="rating")
# # #
# modelExplicit = alsExplicit.fit(training)
# modelImplicit = alsImplicit.fit(training)
# #
# predictionsExplicit = modelExplicit.transform(test)
# predictionsImplicit = modelImplicit.transform(test)
#
# predictionsExplicit.show()
# # model = ALS.train(ratings, 50)
# # userFeatures = model.userFeatures()
# # print (userFeatures.take(2))
