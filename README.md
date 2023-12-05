# bge-api

BGE-large Embeddings api by FastAPI

> Refer to **m3e-large-api**

## Quick Start

### CPU

```sh
docker run -d --name bge-large-api -p 6008:6008 jokerwho/bge-large-api:latest
```

### GPU

> required nvidia-docker2

```sh
docker run -d --name bge-large-api --gpus all -p 6008:6008 jokerwho/bge-large-api:latest
```

## Test

```sh
curl --location --request POST 'http://127.0.0.1/v1/embeddings' \
--header 'Authorization: Bearer sk-aaabbbcccdddeeefffggghhhiiijjjkkk' \
--header 'Content-Type: application/json' \
--data-raw '{
  "model": "bge-large-zh-v1.5",
  "input": ["github是什么"]
}'
```

## Develop

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="BAAI/bge-large-zh-v1.5",cache_dir="./cache", local_dir="models/bge-large-zh-v1.5")
print("======download successful=====")
```

```sh
rm -r ./cache
```

```sh
pip install -r requirements.txt
python main.py
```
