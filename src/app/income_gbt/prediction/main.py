import argparse
import sys

from src.utils.logger import Logger
from src.utils.session import Session
from src.utils.config import Config

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql.functions import when
from pyspark.ml import (
    PipelineModel
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path", help="Path to test data", required=True, type=str
    )
    parser.add_argument(
        "--config_path", help="Path to configuration", type=str, required=True
    )
    parser.add_argument(
        "--model_path", help="Path to model", type=str, required=True
    )
    parser.add_argument(
        "--output_path", help="Path to output", type=str, required=True
    )
    args = parser.parse_args()

    if args.test_data_path is None:
        parser.print_help(sys.stderr)
        exit(0)
    return args


def main(args):
    conf = Config(args.config_path)
    session = Session(conf)
    applog = Logger(session.sc, __name__)
    applog.logger.info("pyspark script logger initialized")

    learning_rate = float(conf.config['MODEL']['learning_rate'])
    applog.logger.info(f'Learning rate: {learning_rate}')
    max_depth = int(conf.config['MODEL']['max_depth'])
    applog.logger.info(f'Max depth: {max_depth}')
    train_rate = float(conf.config['MODEL']['train_rate'])
    applog.logger.info(f'Train rate: {train_rate}')
    seed = int(conf.config['MODEL']['seed'])
    applog.logger.info(f'Seed: {seed}')

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

    test_df = (
            session.spark.read
            .format('csv')
            .option('header', 'true')
            .option('delimiter', ',')
            # .option('inferSchema', 'true')
            .schema(schema)
            .load(args.test_data_path)
            .withColumn('label', when(col('income_level').contains('>50K'), lit(1)).otherwise(lit(0)))
            .drop('education_num', 'income_level', 'label')
            .withColumn('workclass', when(col('workclass') == ' ?', lit('NA')).otherwise(col('workclass')))
            .withColumn('occupation', when(col('occupation') == ' ?', lit('NA')).otherwise(col('occupation')))
            .withColumn('native_country', when(col('native_country') == ' ?', lit('NA')).otherwise(col('native_country')))
        )

    applog.logger.info(args.test_data_path)
    applog.logger.info(test_df.count())
    applog.logger.info(test_df.columns)
    applog.logger.info(test_df.printSchema())

    pipeline_model = PipelineModel.load(args.model_path)
    udf_pos_prob = udf(lambda v: float(v[1]), FloatType())
    test_df_pred = pipeline_model.transform(test_df)
    test_df_pred = test_df_pred.withColumn('prob', udf_pos_prob(col('probability')))
    cols = ['rawPrediction', 'probability', 'prediction', 'prob']
    test_df_pred.select(*cols).write.parquet(args.output_path, mode='overwrite')

    session.sc.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
