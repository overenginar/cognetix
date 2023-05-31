import argparse
import sys

from src.utils.logger import Logger
from src.utils.session import Session

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path", help="Path to train data", required=True, type=str
    )
    parser.add_argument(
        "--test_data_path", help="Path to test data", type=str
    )
    args = parser.parse_args()

    if args.train_data_path is None:
        parser.print_help(sys.stderr)
        exit(0)
    return args


def main(args):
    session = Session()
    applog = Logger(session.sc, __name__)
    applog.logger.info("pyspark script logger initialized")
    # logger.info(conf.config["MODEL"]["max_depth"])

    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", DoubleType(), True),
        StructField("education", StringType(), True),
        StructField("education_num", IntegerType(), True),
        StructField("marital_status", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("relationship", StringType(), True),
        StructField("race", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("capital_gain", DoubleType(), True),
        StructField("capital_loss", DoubleType(), True),
        StructField("hours_per_week", DoubleType(), True),
        StructField("native_country", StringType(), True),
        StructField("income_level", StringType(), True),
    ])
    
    train_df = (
        session.spark.read
        .format('csv')
        .option('header', 'false')
        .option('delimiter', ',')
        #.option('inferSchema', 'true')
        .schema(schema)
        .load(args.train_data_path)
    )

    applog.logger.info(args.train_data_path)
    applog.logger.info(train_df.count())
    applog.logger.info(train_df.columns)
    applog.logger.info(train_df.printSchema())

    if args.test_data_path:
        test_df = (
            session.spark.read
            .format('csv')
            .option('header', 'false')
            .option('delimiter', ',')
            #.option('inferSchema', 'true')
            .schema(schema)
            .load(args.test_data_path)
        )

        applog.logger.info(args.test_data_path)
        applog.logger.info(test_df.count())
        applog.logger.info(test_df.columns)
        applog.logger.info(test_df.printSchema())
    
    session.sc.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
