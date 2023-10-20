from src.utils.config import Config
from pyspark.sql import SparkSession
from pysparkling import H2OContext
import h2o


class Session:
    def __init__(self, conf: Config) -> None:
        self.spark = SparkSession.builder.appName('cognetix-spark-app')
        for k, v in conf.config["SPARK_CONF"].items():
            self.spark = self.spark.config(k, v)
        self.spark = self.spark.getOrCreate()
        self.sc = self.spark.sparkContext
        self.hc = H2OContext.getOrCreate()
        self.h2o_cluster = h2o.cluster()
