import asyncio
import inspect
from typing import Any, AsyncGenerator, Dict, Mapping, Optional, Sequence, Set, Union
import warnings
from autogen_core import CancellationToken
from autogen_core.models import (
    CreateResult,
    ChatCompletionClient,
    RequestUsage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    LLMMessage,
)
from autogen_core.tools import Tool, ToolSchema

from autogen_core.models import (
    ModelCapabilities,
    ModelInfo,
)

from autogen_anthropic_client.config import AnthropicClientConfiguration
from . import _model_info
from typing_extensions import Unpack


from anthropic import AsyncAnthropic

from anthropic.types.completion_create_params import CompletionCreateParamsNonStreaming
from anthropic.types import MessageParam

create_kwargs = set(CompletionCreateParamsNonStreaming.__annotations__.keys())
anthropic_init_kwargs = set(inspect.getfullargspec(AsyncAnthropic.__init__).kwonlyargs)


def to_anthropic_type(message: LLMMessage) -> Sequence[MessageParam]:
    if isinstance(message, str):
        return [MessageParam(role="user", content=message)]
    if isinstance(message, SystemMessage):
        return [MessageParam(role="assistant", content=message.content)]
    if isinstance(message, UserMessage):
        return [MessageParam(role="user", content=message.content)]
    if isinstance(message, AssistantMessage):
        return [MessageParam(role="assistant", content=message.content)]
    raise ValueError(f"Unknown message type: {type(message)}")


class BaseAnthropicChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        create_args: Dict[str, Any],
        model_info: Optional[ModelInfo] = None,
    ):
        self._client = client
        if model_info is None:
            try:
                self._model_info = _model_info.get_info(create_args["model"])
            except KeyError:
                raise ValueError(
                    "model_info is required if model is not a valid Anthropic model"
                )
        else:
            self._model_info = model_info

        self._resolved_model: Optional[str] = None
        if "model" in create_args:
            self._resolved_model = _model_info.resolve_model(create_args["model"])

        self._create_args = create_args
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Dict[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(
                f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}"
            )

        # Copy the create args and overwrite anything in extra_create_args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        anthropic_messages_nested = [to_anthropic_type(message) for message in messages]
        anthropic_messages_flat = [
            item for sublist in anthropic_messages_nested for item in sublist
        ]

        future = asyncio.ensure_future(
            self._client.messages.create(
                messages=anthropic_messages_flat,
                stream=False,
                max_tokens=1000,
                **create_args,
            )
        )

        if cancellation_token is not None:
            cancellation_token.link_future(future)
        result = await future

        # TODO: parse the actual stop reason (result.stop_reason) and convert to FinishReasons
        stop_reason = "stop"
        return CreateResult(
            finish_reason=stop_reason,
            content=result.content[0].text,
            usage=RequestUsage(
                prompt_tokens=result.usage.input_tokens,
                completion_tokens=result.usage.output_tokens,
            ),
            cached=False,  # TODO: can figure it out from usage probably?
            logprobs=None,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        pass

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        pass

    def remaining_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        pass

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        warnings.warn(
            "capabilities is deprecated, use model_info instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info


required_create_args: Set[str] = set(["model"])
disallowed_create_args = set(["stream", "messages", "function_call", "functions", "n"])


def _anthropic_client_from_config(config: Mapping[str, Any]) -> AsyncAnthropic:
    cfg = {k: v for k, v in config.items() if k in anthropic_init_kwargs}
    return AsyncAnthropic(**cfg)


def _create_args_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    create_args = {k: v for k, v in config.items() if k in create_kwargs}
    create_args_keys = set(create_args.keys())
    if not required_create_args.issubset(create_args_keys):
        raise ValueError(
            f"Required create args are missing: {required_create_args - create_args_keys}"
        )
    if disallowed_create_args.intersection(create_args_keys):
        raise ValueError(
            f"Disallowed create args are present: {disallowed_create_args.intersection(create_args_keys)}"
        )
    return create_args


class AnthropicChatCompletionClient(BaseAnthropicChatCompletionClient):
    def __init__(self, **kwargs: Unpack[AnthropicClientConfiguration]):
        if "model" not in kwargs:
            raise ValueError("model is a required argument")

        copied_args = kwargs.copy()

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs("model_info")
            del copied_args["model_info"]

        client = _anthropic_client_from_config(copied_args)
        create_args = _create_args_from_config(copied_args)

        self._raw_config: Dict[str, Any] = kwargs
        super().__init__(client=client, create_args=create_args, model_info=model_info)
