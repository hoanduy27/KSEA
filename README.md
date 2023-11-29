# -SDH-231-BigData

| Service  | Address | Note| 
| ------------- | ------------- | --|
| Kibana  | localhost:5601  ||
| Kafka  | localhost:9092  ||
| Kafka ui | localhost:8080 ||
| Postgresql | localhost:5432 | postgres/postgres|

# Installation

```bash
docker-compose -f docker-compose.yml down -v && docker-compose -f docker-compose.yml up -d
```

# Running flow (overview)
The active learning flow consists of 3 steps:
1. Train initial model with small set of data -> Evaluate test data on initial model

2. Query from data pool to get the most uncertain data
3. Acquire label for the queried data
4. Retrain the model with (inital data + newly queried labeled data) -> Re-evaluate the new model on the test data.

# Running flow (detail)
1. Make sure you successfully install all docker containers (Read #Installation above)
2. Prepare data: Choose between EASY MODE and NORMAL MODE
    - EASY MODE: Nothing to do, we've prepare dummy data for you inside `data/` folder: `training_init.csv` for initial training data, `test.csv` for test data, `training_pool.csv` for unlabaled data to perform Active learning. 
    - NORMAL MODE: Download [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset to `data/`, unzip and rename the csv file to `training.csv`, then:
    ```sh
        # Navigate to root directory of this project (DON'T cd to data/!!!)
        cd </path/to/project-dir>
        # Run the script
        python3 data/split_data.py
    ```
    You can change the size of initial set, test set or pool set before running the script as you wish.

3. Import test data to ElasticSearch using Kafka. Open 2 terminals and run in the following order: 
    
    (First terminal) Create Kafka topic and Consumer
    ```sh
        python3 kafka-ingestion/src/consumer.py 
    ```

    (Second terminal) Create Producer to push data to the above topic
    ```sh
        python3 kafka-ingestion/src/producer.py 
    ```

    Open `localhost:5601` and make sure index `test-kakfa-ingestion-flow` has been created.

4. Import pool data to ElasticSearch
    ```
        python3 spark-ingestion/src/main.py
    ```
    Open `localhost:5601` and make sure index `spark_index` has been created.

5. Perform Active learning
    ```
        python3 -m ml.trainer
    ```

6. Model Performance Visualization
    - Open Kibana at `localhost:5601`
    - Go to **Management** > **Kibana** > **Saved Objects**
    - Click **Import** then upload the file `model_performance_visualization.ndjson`
    