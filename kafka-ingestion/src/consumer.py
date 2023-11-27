from kafka import KafkaConsumer
import json
from elasticsearch import Elasticsearch, helpers


TOPIC = 'test-topic'
ES_INDEX_NAME = 'test-kakfa-ingestion-flow'

 

es = Elasticsearch("http://localhost:9200")

es.indices.create(index=ES_INDEX_NAME, ignore=400)

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='consumer-1',
    enable_auto_commit=True
)

for message in consumer:
  print(message.value['id'])
  es.index(
    index=ES_INDEX_NAME,
    body=message.value,
    doc_type='_doc',
    id = message.value['id']
  )
