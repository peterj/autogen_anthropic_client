from typing import Dict
from autogen_core.models import ModelFamily, ModelInfo

_MODEL_POINTERS = {
    "sonnet": "claude-3-5-sonnet-20241022",
    "haiku": "claude-3-5-haiku-20241022",
    "opus": "claude-3-opus-20240229",
}

_MODEL_INFO: Dict[str, ModelInfo] = {
    "claude-3-5-sonnet-20241022": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        # TODO: Using unknown here for now, need to update core to add a new model family
        "family": ModelFamily.UNKNOWN,
    },
    "claude-3-5-haiku-20241022": {
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
    },
    "claude-3-opus-20240229": {
        "vision": True,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
    },
}


def resolve_model(model: str) -> str:
    if model in _MODEL_POINTERS:
        return _MODEL_POINTERS[model]
    return model

def get_info(model: str) -> ModelInfo:
    resolved_model = resolve_model(model)
    return _MODEL_INFO[resolved_model]
