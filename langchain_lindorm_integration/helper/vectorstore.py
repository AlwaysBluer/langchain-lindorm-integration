from typing import Any

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)


def import_opensearch() -> Any:
    """Import OpenSearch if available, otherwise raise error."""
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return OpenSearch


def import_bulk() -> Any:
    """Import bulk if available, otherwise raise error."""
    try:
        from opensearchpy.helpers import bulk
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return bulk


def import_not_found_error() -> Any:
    """Import not found error if available, otherwise raise error."""
    try:
        from opensearchpy.exceptions import NotFoundError
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return NotFoundError


def get_lindorm_search_client(lindorm_search_url: str, **kwargs: Any) -> Any:
    """
    Get lindorm search client through `opensearchpy` base on the lindorm_search_url,
    otherwise raise error.
    """
    try:
        opensearch = import_opensearch()
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 600
        if kwargs.get("retry_on_timeout") is None:
            kwargs["retry_on_timeout"] = True
        if kwargs.get("max_retries") is None:
            kwargs["max_retries"] = 3
        if kwargs.get("pool_maxsize") is None:
            kwargs["pool_maxsize"] = 20
        client = opensearch(lindorm_search_url, **kwargs)
    except ValueError as e:
        raise ImportError(
            f"Lindorm Search client string provided is not in proper format. "
            f"Got error: {e} "
        )
    return client
