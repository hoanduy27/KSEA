# -SDH-231-BigData

| Service  | Address | Note| 
| ------------- | ------------- | --|
| Kibana  | localhost:5601  ||
| Kafka  | localhost:9092  ||
| Kafka ui | localhost:8080 ||
| Postgresql | localhost:5432 | postgres/postgres|

Install

```bash
docker-compose -f docker-compose.yml down -v && docker-compose -f docker-compose.yml up -d
```