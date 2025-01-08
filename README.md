## Installation

```bash
pip install -U langchain-lindorm-integration
```

## Embeddings

`LindormAIEmbeddings` class exposes embeddings from LindormIntegration.

```python
import os
from langchain_lindorm_integration import LindormAIEmbeddings

query = "What is the meaning of life?"
embedding = LindormAIEmbeddings(
    endpoint=os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>"),
    username=os.environ.get("AI_USERNAME", "root"),
    password=os.environ.get("AI_PASSWORD", "<PASSWORD>"),
    model_name=os.environ.get("AI_DEFAULT_EMBEDDING_MODEL", "bge_m3_model"),
)  # type: ignore[call-arg]
output = embedding.embed_query(query)
embedding.embed_query("What is the meaning of life?")
```
## Rerank
`LindormAIRerank` class exposes rerank from LindormIntegration.

```python
import os
from langchain_core.documents import Document
from langchain_lindorm_integration.reranker import LindormAIRerank

reranker = LindormAIRerank(
    endpoint=os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>"),
    username=os.environ.get("AI_USERNAME", "root"),
    password=os.environ.get("AI_PASSWORD", "<PASSWORD>"),
    model_name=os.environ.get("AI_DEFAULT_EMBEDDING_MODEL", "bge_m3_model"),
    max_workers=5,
    client=None,
)

docs = [
    Document(page_content="量子计算是计算科学的一个前沿领域"),
    Document(page_content="预训练语言模型的发展给文本排序模型带来了新的进展"),
    Document(
        page_content="文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序"
    ),
    Document(page_content="random text for nothing"),
]

for i, doc in enumerate(docs):
    doc.metadata = {"rating": i, "split_setting": str(i % 5)}
    doc.id = str(i)
results = list()
for i in range(10):
    results.append(
        reranker.compress_documents(
            query="什么是文本排序模型",
            documents=docs,
        )
    )


```


## VectorStore

`LindormVectorStore` class exposes vector store from LindormIntegration.

```python
import os
from langchain_lindorm_integration import LindormVectorStore
from langchain_lindorm_integration import LindormAIEmbeddings
from langchain_core.documents import Document

index_name = "langchain_test_index"
dimension = 1024
http_auth = (
    os.environ.get("SEARCH_USERNAME", "root"),
    os.environ.get("SEARCH_PASSWORD", "<PASSWORD>"),
)

def get_default_embedding():
    return LindormAIEmbeddings(
    endpoint=os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>"),
    username=os.environ.get("AI_USERNAME", "root"),
    password=os.environ.get("AI_PASSWORD", "<PASSWORD>"),
    model_name=os.environ.get("AI_DEFAULT_EMBEDDING_MODEL", "bge_m3_model"),
)  

BUILD_INDEX_PARAMS = {
    "lindorm_search_url": os.environ.get("SEARCH_ENDPOINT", "<SEARCH_ENDPOINT>"),
    "dimension": dimension,
    "embedding": get_default_embedding(),
    "http_auth": http_auth,
    "index_name": index_name,
    "use_ssl": False,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "bulk_size": 500,
    "timeout": 60,
    "max_retries": 3,
    "retry_on_timeout": True,
    "embed_thread_num": 2,
    "write_thread_num": 5,
    "pool_maxsize": 20,
    "engine": "lvector",
    "space_type": "l2",
}

vectorstore = LindormVectorStore(**BUILD_INDEX_PARAMS)
original_documents = [
    Document(page_content="foo", metadata={"id": 1}),
    Document(page_content="bar", metadata={"id": 2}),
]
ids = vectorstore.add_documents(original_documents)
documents = vectorstore.similarity_search("bar", k=2)


```

## ByteStore

```python
import os
from langchain_lindorm_integration import LindormByteStore

http_auth = (
    os.environ.get("SEARCH_USERNAME", "root"),
    os.environ.get("SEARCH_PASSWORD", "<PASSWORD>"),
)
index_name = "langchain_test_index"

DEFAULT_BUILD_PARAMS = {
    "lindorm_search_url": os.environ.get("SEARCH_ENDPOINT", "<SEARCH_ENDPOINT>"),
    "http_auth": http_auth,
    "index_name": index_name,
    "use_ssl": False,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "bulk_size": 10000,
    "timeout": 30
}

bytestore = LindormByteStore(**DEFAULT_BUILD_PARAMS)
try:
    bytestore.mdelete(["k1", "k2"])
except:
    pass

result = bytestore.mget(["k1", "k2"])
print(result)

bytestore.mset([("k1", b"v1"), ("k2", b"v2")])
result = bytestore.mget(["k1", "k2"])
print(result)
assert result == [b"v1", b"v2"]

bytestore.mdelete(["k1", "k2"])
result = bytestore.mget(["k1", "k2"])
print(result)
assert result == [None, None]

bytestore.client.indices.delete(index=index_name)

```

## Build from Source code
```shell
poetry build
```