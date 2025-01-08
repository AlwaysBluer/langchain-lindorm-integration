"""Test Lindorm Search ByteStore."""
import os

from langchain_lindorm_integration.byte_store import LindormByteStore


class Config:
    SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT", "SEARCH_ENDPOINT")
    SEARCH_USERNAME = os.environ.get("SEARCH_USERNAME", "root")
    SEARCH_PWD = os.environ.get("SEARCH_PWD", "<PASSWORD>")


DEFAULT_BUILD_PARAMS = {
    "lindorm_search_url": Config.SEARCH_ENDPOINT,
    "http_auth": (Config.SEARCH_USERNAME, Config.SEARCH_PWD),
    "use_ssl": False,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "bulk_size": 10000,
    "timeout": 30
}


def test_lindorm_search_bytestore(index_name="bytestore_test"):
    DEFAULT_BUILD_PARAMS["index_name"] = index_name
    bytestore = LindormByteStore(**DEFAULT_BUILD_PARAMS)

    try:
        bytestore.mdelete(["k1", "k2"])
    except:
        pass

    result = bytestore.mget(["k1", "k2"])
    print(result)
    assert result == [None, None]

    bytestore.mset([("k1", b"v1"), ("k2", b"v2")])
    result = bytestore.mget(["k1", "k2"])
    print(result)
    assert result == [b"v1", b"v2"]

    bytestore.mdelete(["k1", "k2"])
    result = bytestore.mget(["k1", "k2"])
    print(result)
    assert result == [None, None]

    bytestore.client.indices.delete(index=index_name)


def test_lindorm_search_bytestore_yield_keys(index_name="bytestore_test"):
    DEFAULT_BUILD_PARAMS["index_name"] = index_name
    bytestore = LindormByteStore(**DEFAULT_BUILD_PARAMS)

    bytestore.mset([("b", b"v1"),
                    ("by", b"v2"),
                    ("byte", b"v3"),
                    ("bytestore", b"v4"),
                    ("s", b"v5"),
                    ("st", b"v6"),
                    ("sto", b"v7"),
                    ("stor", b"v8"),
                    ("store", b"v9"),
                    ])
    bytestore.client.indices.refresh(index=index_name)

    prefix = "b"
    keys = list(bytestore.yield_keys(prefix=prefix))
    print(keys)
    assert len(keys) == 4

    prefix = "sto"
    keys = list(bytestore.yield_keys(prefix=prefix))
    print(keys)
    assert len(keys) == 3

    bytestore.client.indices.delete(index=index_name)


if __name__ == "__main__":
    test_lindorm_search_bytestore()
    test_lindorm_search_bytestore_yield_keys()
