from datetime import datetime, timedelta
import pendulum
from airflow import DAG
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from airflow.models import Variable
from airflow.operators.bash import BashOperator

local_tz = pendulum.timezone("Europe/London")

default_args = {
    'owner': 'overenginar',
    'depends_on_past': False,
    'start_date': datetime(2020, 10, 10, tzinfo=local_tz),
    'email': ['overenginar@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}
dag = DAG(dag_id='income_pred_xgb_dag',
          default_args=default_args,
          catchup=False,
          schedule_interval="0 * * * *")

pyspark_app_home = Variable.get("PYSPARK_APP_HOME")

create_dir = BashOperator(
    task_id="create_dir",
    bash_command="mkdir -p /home/jovyan/af_data",
    dag=dag
    )
download_train_data = BashOperator(
    task_id="download_train_data",
    bash_command="wget -O /home/jovyan/af_data/census-train.csv http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
    dag=dag
)

download_test_data = BashOperator(
    task_id="download_test_data",
    bash_command="wget -O /home/jovyan/af_data/census-test.csv http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", 
    dag=dag
)

train_automl = SparkSubmitOperator(
    task_id='train_automl',
    conn_id='spark_conn',
    application=f'{pyspark_app_home}/income_train_automl.py',
    total_executor_cores=4,
    executor_cores=2,
    executor_memory='1g',
    driver_memory='1g',
    name='train_automl',
    execution_timeout=timedelta(minutes=10),
    dag=dag
    )

pred_automl = SparkSubmitOperator(
    task_id='pred_automl',
    conn_id='spark_conn',
    application=f'{pyspark_app_home}/income_pred_automl.py',
    total_executor_cores=4,
    executor_cores=2,
    executor_memory='1g',
    driver_memory='1g',
    name='pred_automl',
    execution_timeout=timedelta(minutes=10),
    dag=dag
    )

create_dir >> [download_train_data, download_test_data] >> train_automl >> pred_automl
