"""Test LindormIntegration embeddings."""
import os
from typing import Type

from langchain_lindorm_integration.embeddings import LindormAIEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class Config:
    AI_ENDPOINT = os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PASSWORD", "<PASSWORD>")

    AI_DEFAULT_EMBEDDING_MODEL = "bge_m3_model"


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[LindormAIEmbeddings]:
        return LindormAIEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "endpoint": Config.AI_ENDPOINT,
            "username": Config.AI_USERNAME,
            "password": Config.AI_PWD,
            "model_name": Config.AI_DEFAULT_EMBEDDING_MODEL,
        }
