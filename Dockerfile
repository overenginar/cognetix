FROM jupyter/pyspark-notebook:spark-3.1.2

USER root
RUN sudo apt-get update -y && sudo apt-get install zip unzip -y

USER jovyan
COPY . .

RUN pip install h2o-pysparkling-3.1==3.40.0.4.post1
RUN pip install h2o==3.40.0.4
RUN pip install "apache-airflow[celery]==2.7.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.8.txt"
RUN pip install apache-airflow-providers-apache-spark==4.1.5

EXPOSE 8080/tcp

ENV AIRFLOW_HOME=~/airflow

# RUN airflow db init

# RUN airflow users create \
#    --username admin \
#    --password admin \
#    --firstname Ali \
#    --lastname Cabukel \
#    --role Admin \
#    --email acabukel@airflow.org

# RUN airflow sync-perm

# RUN airflow webserver --port 8080

# RUN airflow scheduler

CMD airflow standalone
