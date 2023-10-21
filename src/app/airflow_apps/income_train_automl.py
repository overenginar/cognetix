from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql.functions import lit
import logging
from pysparkling import H2OContext
from h2o.automl import H2OAutoML
import h2o


def main():
    data_path = '/home/jovyan/af_data/census-train.csv'
    model_path = '/home/jovyan/af_output/models'

    logger = logging.getLogger('py4j')
    logger.info("My test info statement")
    logger.info("This is a log message")
    spark = (
        SparkSession
        .builder
        .master('local[*]')
        .appName('income_pred_xgb')
        .getOrCreate()
    )
    # spark.sparkContext.setLogLevel("ERROR")
    # sc = spark.sparkContext
    hc = H2OContext.getOrCreate()

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
        spark.read
        .format('csv')
        .option('header', 'false')
        .option('delimiter', ',')
        .schema(schema)
        .load(data_path)
        .drop('education_num')
        .withColumn('label', when(col('income_level').contains('>50K'), lit(1)).otherwise(lit(0)))
        .drop('income_level')
        .withColumn('workclass', when(col('workclass') == ' ?', lit('NA')).otherwise(col('workclass')))
        .withColumn('occupation', when(col('occupation') == ' ?', lit('NA')).otherwise(col('occupation')))
        .withColumn('native_country', when(col('native_country') == ' ?', lit('NA')).otherwise(col('native_country')))
    )

    train_hdf = hc.asH2OFrame(train_df)
    train_hdf['label'] = train_hdf['label'].asfactor()
    train_hdf['workclass'] = train_hdf['workclass'].asfactor()
    train_hdf['education'] = train_hdf['education'].asfactor()
    train_hdf['marital_status'] = train_hdf['marital_status'].asfactor()
    train_hdf['occupation'] = train_hdf['occupation'].asfactor()
    train_hdf['relationship'] = train_hdf['relationship'].asfactor()
    train_hdf['race'] = train_hdf['race'].asfactor()
    train_hdf['sex'] = train_hdf['sex'].asfactor()
    train_hdf['native_country'] = train_hdf['native_country'].asfactor()

    x = [a for a in list(train_hdf.columns) if a != 'label']
    y = 'label'
    model = H2OAutoML(
        max_models=5,
        max_runtime_secs=60,
        include_algos=['XGBoost', 'GBM'],
        seed=42,
        verbosity='debug',
        keep_cross_validation_predictions=True,
        nfolds=5
    )
    model.train(x=x, y=y, training_frame=train_hdf)
    best_model = model.leader
    best_model.model_id = 'automl_leader'
    h2o.save_model(best_model, model_path, force=True)


if __name__ == '__main__':
    main()
