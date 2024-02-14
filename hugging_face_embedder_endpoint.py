from cat.mad_hatter.decorators import hook
from typing import List, Optional, Type
from pydantic import ConfigDict, SecretStr
from cat.factory.embedder import EmbedderSettings
import json
from typing import List
from langchain.embeddings.base import Embeddings
import httpx
from typing import List


class CustomHuggingFaceEmbeddings(Embeddings):
    
    def __init__(self, huggingface_api_key, huggingface_endpoint):
        self.url = huggingface_endpoint
        self.token = huggingface_api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"inputs": texts})
        headers = {"Authorization": f"Bearer {self.token}",
                   "Content-Type": "application/json"}
        ret = httpx.post(self.url, data=payload, headers=headers, timeout=None)
        ret.raise_for_status()
        return  [e for e in ret.json()]
    
    def embed_query(self, text: str) -> List[float]:
        payload = json.dumps({"inputs": text})
        headers = {"Authorization": f"Bearer {self.token}",
                   "Content-Type": "application/json"}
    
        ret = httpx.post(self.url, data=payload, headers=headers, timeout=None)
        ret.raise_for_status()
        return ret.json()[0]


class HFEmbedder(EmbedderSettings):
    huggingface_api_key: Optional[SecretStr]
    huggingface_endpoint: Optional[SecretStr]
    _pyclass: Type = CustomHuggingFaceEmbeddings

    model_config = ConfigDict(
        json_schema_extra = {
            "humanReadableName": "HuggingFace Embedder",
            "description": "Configuration for HuggingFace embeddings",
            "link": "https://huggingface.co/blog/inference-endpoints-embeddings",
        }
    )

@hook
def factory_allowed_embedders(allowed, cat) -> List:
    allowed.append(HFEmbedder)
    return allowed
