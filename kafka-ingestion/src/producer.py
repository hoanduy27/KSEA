from kafka import KafkaProducer
import json
import pandas as pd

TOPIC = 'test-topic'

producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

data = pd.read_csv('data/test.csv')

json_array = data.to_dict('records')

for event in json_array:
  producer.send(TOPIC, event)

producer.flush()

producer.close()

print('done')