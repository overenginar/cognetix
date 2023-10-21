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
import h2o


def main():
    data_path = '/home/jovyan/af_data/census-test.csv'
    model_path = '/home/jovyan/af_output/models'
    pred_path = '/home/jovyan/af_output/predictions/automl_leader_pred'
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

    test_df = (
        spark.read
        .format('csv')
        .option('header', 'true')
        .option('delimiter', ',')
        .schema(schema)
        .load(data_path)
        .drop('education_num')
        .withColumn('label', when(col('income_level').contains('>50K'), lit(1)).otherwise(lit(0)))
        .drop('income_level', 'label')
        .withColumn('workclass', when(col('workclass') == ' ?', lit('NA')).otherwise(col('workclass')))
        .withColumn('occupation', when(col('occupation') == ' ?', lit('NA')).otherwise(col('occupation')))
        .withColumn('native_country', when(col('native_country') == ' ?', lit('NA')).otherwise(col('native_country')))
    )
    test_hdf = hc.asH2OFrame(test_df)
    test_hdf['workclass'] = test_hdf['workclass'].asfactor()
    test_hdf['education'] = test_hdf['education'].asfactor()
    test_hdf['marital_status'] = test_hdf['marital_status'].asfactor()
    test_hdf['occupation'] = test_hdf['occupation'].asfactor()
    test_hdf['relationship'] = test_hdf['relationship'].asfactor()
    test_hdf['race'] = test_hdf['race'].asfactor()
    test_hdf['sex'] = test_hdf['sex'].asfactor()
    test_hdf['native_country'] = test_hdf['native_country'].asfactor()
    model = h2o.load_model(model_path + '/automl_leader')
    df_predict = model.predict(test_hdf)
    df_predict = df_predict.cbind(test_hdf)
    (
        hc.asSparkFrame(df_predict)
        .write.parquet(pred_path, mode='overwrite')
    )


if __name__ == '__main__':
    main()
