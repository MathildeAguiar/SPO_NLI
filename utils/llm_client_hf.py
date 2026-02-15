import asyncio
import re
from enum import Enum
from typing import Any, List, Optional

from llm.async_llm_hf import AsyncLLM, LLMsConfig
from loguru import logger


class RequestType(Enum):
    OPTIMIZE = "optimize"
    EVALUATE = "evaluate"
    EXECUTE = "execute"


class SPO_LLM:
    _instance: Optional["SPO_LLM"] = None

    def __init__(
        self,
        optimize_kwargs: Optional[dict] = None,
        evaluate_kwargs: Optional[dict] = None,
        execute_kwargs: Optional[dict] = None,
        mode: str = "base_model"
    ) -> None:

        optimize_kwargs = optimize_kwargs or {}
        evaluate_kwargs = evaluate_kwargs or {}
        execute_kwargs = execute_kwargs or {}

        self.evaluate_llm = AsyncLLM(config=self._load_llm_config(evaluate_kwargs))
        self.optimize_llm = AsyncLLM(config=self._load_llm_config(optimize_kwargs))
        self.execute_llm = AsyncLLM(config=self._load_llm_config(execute_kwargs), mode=mode)

    def _load_llm_config(self, kwargs: dict) -> Any:
        model_name = kwargs.get("model")
        if not model_name:
            raise ValueError("'model' parameter is required")

        try:
            # Load config directly by requested model name
            model_config = LLMsConfig.default().get(model_name)

            # Clone config so we don't mutate shared default
            config = type(model_config)(model_config.__dict__.copy())

            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            return config

        except Exception as e:
            raise ValueError(f"Error loading configuration for model '{model_name}': {str(e)}")

    async def responser(self, request_type: RequestType, messages: List[dict]) -> str:
        llm_mapping = {
            RequestType.OPTIMIZE: self.optimize_llm,
            RequestType.EVALUATE: self.evaluate_llm,
            RequestType.EXECUTE: self.execute_llm,
        }

        llm = llm_mapping.get(request_type)
        if not llm:
            raise ValueError(
                f"Invalid request type. Valid types: {', '.join([t.value for t in RequestType])}"
            )

        response = await llm(messages)
        return response

    @classmethod
    def initialize(cls, optimize_kwargs: dict, evaluate_kwargs: dict, execute_kwargs: dict, mode: str) -> None:
        cls._instance = cls(optimize_kwargs, evaluate_kwargs, execute_kwargs, mode)

    @classmethod
    def get_instance(cls) -> "SPO_LLM":
        if cls._instance is None:
            raise RuntimeError("SPO_LLM not initialized. Call initialize() first.")
        return cls._instance


def extract_content(xml_string: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, xml_string, re.DOTALL)
    return match.group(1).strip() if match else None
