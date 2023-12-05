# bge-api

BGE-large Embeddings api by FastAPI

> Refer to **m3e-large-api**

```sh
docker run -d --name bge-large-api -p 6008:6008 jokerwho/bge-large-api:latest
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
