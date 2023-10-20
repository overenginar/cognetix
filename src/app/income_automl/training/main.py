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
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql.functions import lit
from h2o.automl import H2OAutoML


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path", help="Path to train data", required=True, type=str
    )
    parser.add_argument(
        "--config_path", help="Path to configuration", type=str, required=True
    )
    parser.add_argument(
        "--output_path", help="Path to output", type=str, required=True
    )
    args = parser.parse_args()

    if args.train_data_path is None:
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

    train_df = (
        session.spark.read
        .format('csv')
        .option('header', 'false')
        .option('delimiter', ',')
        # .option('inferSchema', 'true')
        .schema(schema)
        .load(args.train_data_path)
        .drop('education_num')
        .withColumn('label', when(col('income_level').contains('>50K'), lit(1)).otherwise(lit(0)))
        .drop('income_level')
        .withColumn('workclass', when(col('workclass') == ' ?', lit('NA')).otherwise(col('workclass')))
        .withColumn('occupation', when(col('occupation') == ' ?', lit('NA')).otherwise(col('occupation')))
        .withColumn('native_country', when(col('native_country') == ' ?', lit('NA')).otherwise(col('native_country')))
    )

    applog.logger.info(args.train_data_path)
    applog.logger.info(train_df.count())
    applog.logger.info(train_df.columns)
    applog.logger.info(train_df.printSchema())

    train_hdf = session.hc.asH2OFrame(train_df)
    # train_hdf['cv_fold'] = train_hdf['label'].kfold_column(n_folds=5, seed=seed)
    # train_hdf['split'] = train_hdf['label'].stratified_split(test_frac=0.6, seed=seed)
    # df_train, df_val = df.split_frame([0.6], seed=seed)
    train_hdf['label'] = train_hdf['label'].asfactor()
    train_hdf['workclass'] = train_hdf['workclass'].asfactor()
    train_hdf['education'] = train_hdf['education'].asfactor()
    train_hdf['marital_status'] = train_hdf['marital_status'].asfactor()
    train_hdf['occupation'] = train_hdf['occupation'].asfactor()
    train_hdf['relationship'] = train_hdf['relationship'].asfactor()
    train_hdf['race'] = train_hdf['race'].asfactor()
    train_hdf['sex'] = train_hdf['sex'].asfactor()
    train_hdf['native_country'] = train_hdf['native_country'].asfactor()
    cols = list(train_hdf.columns)
    x = [a for a in cols if a != 'label']
    y = 'label'
    model = H2OAutoML(
        # max_models=250,
        # max_runtime_secs=3600,
        max_models=5,
        max_runtime_secs=180,
        include_algos=['XGBoost', 'GBM'],
        seed=seed,
        verbosity='debug',
        keep_cross_validation_predictions=True,
        nfolds=5
    )
    model.train(x=x, y=y, training_frame=train_hdf)
    best_model = model.leader
    applog.logger.info(f'Model leaderboard: {model.leaderboard.as_data_frame()}')
    applog.logger.info(f'Actual params: {best_model.actual_params}')
    applog.logger.info(f'Train performance: {best_model.model_performance(train=True)}')
    applog.logger.info(f'CV performance: {best_model.model_performance(xval=True)}')
    applog.logger.info(f"Label distribution: {train_hdf['label'].table().as_data_frame()}")
    # model_perf_test = best_model.model_performance(valid=True)
    best_model.model_id = 'automl_leader'
    best_model.save_mojo(args.output_path, force=True)
    session.sc.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
