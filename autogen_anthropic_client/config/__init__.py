from typing import Optional
from typing_extensions import TypedDict
from autogen_core.models import  ModelInfo


class AnthropicClientConfiguration(TypedDict, total=False):
    model: str
    temperature: Optional[float]
    top_k: Optional[float]
    top_p: Optional[float]
    max_tokens: int
    api_key: str
    base_url: str | None = None
    model_info: ModelInfo | None = None
    timeout: float | None = None