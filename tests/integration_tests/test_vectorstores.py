import logging
import os
from importlib import util
from typing import AsyncGenerator, Generator, Any

import pytest
from langchain_tests.integration_tests.vectorstores import EMBEDDING_SIZE

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
    SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT", "SEARCH_ENDPOINT")
    SEARCH_USERNAME = os.environ.get("SEARCH_USERNAME", "root")
    SEARCH_PWD = os.environ.get("SEARCH_PWD", "<PASSWORD>")


logger = logging.getLogger(__name__)


index_name = "langchain_test_index"

BUILD_INDEX_PARAMS = {
    "lindorm_search_url": Config.SEARCH_ENDPOINT,
    "dimension": EMBEDDING_SIZE,
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
