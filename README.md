# langchain-lindorm-integration

This package contains the LangChain integration with LindormIntegration

## Installation

```bash
pip install -U langchain-lindorm-integration
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatLindormIntegration` class exposes chat models from LindormIntegration.

```python
from langchain_lindorm_integration import ChatLindormIntegration

llm = ChatLindormIntegration()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`LindormIntegrationEmbeddings` class exposes embeddings from LindormIntegration.

```python
from langchain_lindorm_integration import LindormIntegrationEmbeddings

embeddings = LindormIntegrationEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`LindormIntegrationLLM` class exposes LLMs from LindormIntegration.

```python
from langchain_lindorm_integration import LindormIntegrationLLM

llm = LindormIntegrationLLM()
llm.invoke("The meaning of life is")
```
