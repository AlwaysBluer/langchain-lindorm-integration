import logging
import os
from importlib import util
from typing import AsyncGenerator, Generator, Any

import pytest
from langchain_lindorm_integration.vectorstores import LindormVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)


def _get_opensearch_scan() -> Any:
    if util.find_spec("opensearchpy.helpers"):
        from opensearchpy.helpers import scan

        return scan  # 返回 bulk 函数的引用
    else:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)


class Config:
    AI_LLM_ENDPOINT = os.environ.get("AI_LLM_ENDPOINT", "<LLM_ENDPOINT>")
    AI_EMB_ENDPOINT = os.environ.get("AI_EMB_ENDPOINT", "<EMB_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PWD", "<PASSWORD>")

    AI_DEFAULT_RERANK_MODEL = "rerank_bge_v2_m3"
    AI_DEFAULT_EMBEDDING_MODEL = "bge_m3_model"
    SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT", "SEARCH_ENDPOINT")
    SEARCH_USERNAME = os.environ.get("SEARCH_USERNAME", "root")
    SEARCH_PWD = os.environ.get("SEARCH_PWD", "<PASSWORD>")


logger = logging.getLogger(__name__)


# def get_default_embedding() -> Any:
#     embedding = LindormAIEmbeddings(
#         endpoint=Config.AI_LLM_ENDPOINT,
#         username=Config.AI_USERNAME,
#         password=Config.AI_PWD,
#         model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
#         client=None,
#     )
#     return embedding

# def get_default_embedding() -> Any:


index_name = "langchain_test_index"

BUILD_INDEX_PARAMS = {
    "lindorm_search_url": Config.SEARCH_ENDPOINT,
    # default params
    # "embedding": get_default_embedding(),
    "http_auth": (Config.SEARCH_USERNAME, Config.SEARCH_PWD),
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

BUILD_INDEX = True


class TestLindormIntegrationVectorStoreSync(VectorStoreIntegrationTests):

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        BUILD_INDEX_PARAMS["embedding"] = self.get_embeddings()
        store = LindormVectorStore(**BUILD_INDEX_PARAMS)
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            store.delete_index(index_name)
            pass
