from pyspark import SparkContext

from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
SparkContext.creat

logFile = "/usr/spark-1.6.0-bin-hadoop1/README.md"
sc = SparkContext("local","Simple App")
logData = sc.textFile(logFile).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
print("Lines with a: %i,lines with b: %i"%(numAs,numBs))