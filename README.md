# cognetix

### Building Docker image

```shell
docker build -t overenginar/cognetix:0.0.1 .
```

### Creating container

```shell
docker run --name cognetix-dev -it -d -p 8901:8888 -p 8902:4040 -p 8903:8080 --user root -e GRANT_SUDO=yes overenginar/cognetix:0.0.1
```

### Check airflow admin password

```shell
# username: admin
docker exec -it cognetix-dev cat airflow/standalone_admin_password.txt
```

### Testing airflow example DAG

```shell
docker exec -it cognetix-dev airflow tasks test example_bash_operator runme_0 2015-01-01

docker exec -it cognetix-dev airflow dags backfill example_bash_operator \
    --start-date 2015-01-01 \
    --end-date 2015-01-02
```

### Getting census income data

```shell
docker exec -it cognetix-dev mkdir -p data
docker exec -it cognetix-dev wget -O data/census-train.csv http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
docker exec -it cognetix-dev wget -O data/census-test.csv http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test 
```

### spark-submit an example

```shell
docker exec -it cognetix-dev zip -r /tmp/cognetix.zip src -x ".*"
docker exec -it cognetix-dev ls /tmp/cognetix.zip
docker exec -it cognetix-dev spark-submit --py-files config.ini,/tmp/cognetix.zip src/app/main.py --train_data_path data/census-train.csv --test_data_path data/census-test.csv
```

### Running jupyter-lab for testing

```shell
#welcome1
docker exec -it -d cognetix-dev jupyter-lab --allow-root
```

### Copy source code

```shell
docker cp . cognetix-dev:/home/jovyan
```

### Copy data

```shell
docker cp data cognetix-dev:/home/jovyan/data
```
