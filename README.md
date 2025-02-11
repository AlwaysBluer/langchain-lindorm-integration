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

### BUILD_INDEX_PARAMS Configuration
you can find more detailed information on [this page](https://help.aliyun.com/document_detail/2773371.html?spm=a2c4g.11186623.help-menu-172543.d_2_5_1_0_0.7c5c3491qd8H8L#3d63548e78dm4)

#### Connection Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| lindorm_search_url | str | Lindorm search URL. Format: "http://ld-bp106782jm96****-proxy-search-vpc.lindorm.aliyuncs.com:30070" | - |
| http_auth | tuple | (username, password) for authentication | - |
| use_ssl | bool | Enable SSL | False |
| verify_certs | bool | Verify SSL certificates | False |
| ssl_assert_hostname | bool | Assert hostname in SSL connections | False |
| ssl_show_warn | bool | Show SSL warnings | False |
| timeout | int | Timeout for HTTP requests (in seconds) | - |
| max_retries | int | Maximum number of retry attempts for HTTP requests | - |
| retry_on_timeout | bool | Retry on timeout | True |

#### Index Configuration

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| index_name | str | Lindorm index name | - |
| embedding | Embeddings | Langchain Embeddings object | - |
| vector_field | str | Lindorm vector field name | "vector_field" |
| shards | int | Number of shards for Lindorm index | 2 |
| method_name | str | Method name for Lindorm index (options: hnsw, ivfpq) | - |
| data_type | str | Data type for Lindorm index (options: float, float16, sparse_vector) | float |
| space_type | str | Space type for Lindorm index (options: l2, innerproduct, cosinesimil) | cosinesimil |
| engine | str | Should be 'lvector' | lvector |

#### HNSW Specific Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| hnsw_m | int | HNSW M parameter | 24 |
| hnsw_ef_construction | int | HNSW ef construction parameter | 500 |

#### IVFPQ Specific Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| ivfpq_m | int | IVFPQ M parameter | Equal to vector dimension |
| nlist | int | Number of clusters for IVFPQ | 1000 |
| centroids_use_hnsw | bool | Use HNSW for centroids | True if nlist >= 5000 |
| centroids_hnsw_m | int | HNSW M parameter for centroids | 24 |
| centroids_hnsw_ef_construct | int | HNSW ef construction parameter for centroids | 500 |
| centroids_hnsw_ef_search | int | HNSW ef search parameter for centroids | 100 |

### Performance Tuning

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| bulk_size | int | Batch size for bulk insert | - |
| write_thread_num | int | Number of threads for bulk write operations | - |
| embed_thread_num | int | Number of threads for embedding operations | - |

[The file](tests/integration_tests/test_custom_vectorstore.py) contains more complicated examples in the UGC scenario.

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