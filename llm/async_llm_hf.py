from huggingface_hub import AsyncInferenceClient
import yaml
from pathlib import Path
from typing import Dict, Optional, Any

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", None)
        self.top_p = config.get("top_p", 1)


class LLMsConfig:
    _instance = None
    _default_config = None

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.configs = config_dict or {}

    @classmethod
    def default(cls):
        if cls._default_config is None:
            config_paths = [
                Path("config/config2.example_hf.yaml"),
                Path("config2.example_hf.yaml"),
                Path("./config/config2.example_hf.yaml")
            ]

            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break

            if config_file is None:
                raise FileNotFoundError("No default configuration file found")

            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if 'models' in config_data:
                config_data = config_data['models']

            cls._default_config = cls(config_data)

        return cls._default_config

    def get(self, llm_name: str) -> LLMConfig:
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")

        config = self.configs[llm_name]

        llm_config = {
            "model": llm_name,
            "temperature": config.get("temperature", 1),
            "key": config.get("api_key"),
            "base_url": config.get("base_url"),
            "top_p": config.get("top_p", 1)
        }

        return LLMConfig(llm_config)

    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        self.configs[name] = config

    def get_all_names(self) -> list:
        return list(self.configs.keys())


class ModelPricing:
    PRICES = {}

    @classmethod
    def get_price(cls, model_name, token_type):
        return 0


class TokenUsageTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []

    def add_usage(self, model, input_tokens, output_tokens):
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
            "prices": {"input_price": 0, "output_price": 0}
        }

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.usage_history.append(usage_record)

        return usage_record

    def get_summary(self):
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": 0,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }


class AsyncLLM:
    def __init__(self, config, system_msg: str = None, mode: str = "base_model"):

        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)

        self.config = config

        self.client = AsyncInferenceClient(
            model=self.config.model,
            token=self.config.key
        )

        self.sys_msg = system_msg
        self.mode = mode
        self.usage_tracker = TokenUsageTracker()

        self.tokenizer = None
        if AutoTokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            except Exception:
                pass

    def _format_messages(self, messages):
        formatted = ""
        if self.sys_msg:
            formatted += f"System: {self.sys_msg}\n"

        for m in messages:
            formatted += f"{m['role'].capitalize()}: {m['content']}\n"

        formatted += "Assistant:"
        return formatted

    def _count_tokens(self, text):
        if not self.tokenizer:
            return len(text.split())
        return len(self.tokenizer.encode(text))

    async def __call__(self, messages):

        prompt = self._format_messages(messages)

        result = await self.client.text_generation(
            prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=1024
        )

        response_text = result.strip()

        input_tokens = self._count_tokens(prompt)
        output_tokens = self._count_tokens(response_text)

        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )

        if self.mode == "reasoning_model":
            ret = f"Thought:\n{response_text}\nAnswer:\n{response_text}"
        else:
            ret = response_text

        print(f"Token usage: {input_tokens} + {output_tokens}")
        return ret

    def get_usage_summary(self):
        return self.usage_tracker.get_summary()


def create_llm_instance(llm_config):

    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)

    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)

    elif isinstance(llm_config, dict):
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)

    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
