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
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    OneHotEncoder,
    Imputer,
)
from pyspark.ml import (
    Pipeline,
    PipelineModel
)
from pyspark.ml.classification import(
    GBTClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator


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

    train_df, val_df = train_df.randomSplit([train_rate, 1-train_rate], seed=seed)
    applog.logger.info(f'Train split size: {train_df.count()}')
    applog.logger.info(f'Validation split size: {val_df.count()}')
    # train_df.sampleBy("income_level", {}, seed)

    applog.logger.info(f"Label Distribution: {train_df.groupBy('label').count().toPandas()}")
    # applog.logger.info(f'Describe DF: {train_df.describe().toPandas()}')
    # applog.logger.info(f'Summary DF: {train_df.summary().toPandas()}')
    cols_to_impute = ['fnlwgt', 'age', 'capital_gain', 'capital_loss', 'hours_per_week']
    cat_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    imputed_cols = [f'{x}_IMPUTED' for x in cols_to_impute]
    imputer = Imputer(strategy='mean', inputCols=cols_to_impute, outputCols=imputed_cols)
    string_indexers = []
    ohe_indexers = []
    for cat_col in cat_cols:
        si = StringIndexer(inputCol=cat_col, outputCol=f'{cat_col}_idx').setHandleInvalid('keep')
        enc = OneHotEncoder(inputCols=[si.getOutputCol()], outputCols=[f'{cat_col}_vec'])
        string_indexers.append(si)
        ohe_indexers.append(enc)

    assembler_cols = [f'{c}_vec' for c in cat_cols] + imputed_cols
    vector_assembler = VectorAssembler(inputCols=assembler_cols, outputCol='features')
    rf = GBTClassifier(labelCol='label', featuresCol='features')
    rf_stages = [imputer] + string_indexers + ohe_indexers + [vector_assembler] + [rf]
    pipeline = Pipeline().setStages(rf_stages)
    rf_model = pipeline.fit(train_df)
    rf_model.write().overwrite().save(args.output_path)

    pipeline_model = PipelineModel.load(args.output_path)
    val_df_pred = pipeline_model.transform(val_df)
    evaluator = BinaryClassificationEvaluator()
    applog.logger.info(f'Metric name: {evaluator.getMetricName()}')
    applog.logger.info(f'Metric value: {evaluator.evaluate(val_df_pred)}')

    session.sc.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
