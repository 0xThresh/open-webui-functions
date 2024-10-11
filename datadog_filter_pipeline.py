"""
title: DataDog Filter Pipeline
author: James W. (0xThresh)
author_url: https://github.com/0xThresh
funding_url: https://github.com/open-webui
version: 1.1
requirements: ddtrace
"""

# You can read more about the full configuration here: https://blog.opensourceai.dev/monitor-open-webui-with-datadog-llm-observability-620ef3a598c6
from typing import Optional
from open_webui.utils.misc import get_last_user_message, get_last_assistant_message
from pydantic import Field, BaseModel
from ddtrace.llmobs import LLMObs


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        DD_API_KEY: str = Field(
            description="DataDog API key"
        )
        DD_SITE: str = Field(
            default="datadoghq.com",
            description="Your DataDog site. Set if your account is hosted in a specific region.",
        )
        ML_APP: str = Field(
            default="Open WebUI",
            description="Name of the app shown in DataDog LLM traces",
        )
        pass

    def __init__(self):
        self.type = "filter"
        self.name = "DataDog Filter"

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()

        # DataDog LLMOBS docs: https://docs.datadoghq.com/tracing/llm_observability/sdk/
        self.LLMObs = LLMObs()
        self.llm_span = None
        self.chat_generations = {}
        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:

        self.LLMObs.enable(
            ml_app=self.valves.ML_APP,
            api_key=self.valves.DD_API_KEY,
            site=self.valves.DD_SITE,
            agentless_enabled=True,
            integrations_enabled=True,
            env="test",
            service="test",
        )

        self.llm_span = self.LLMObs.llm(
            model_name=body["model"],
            name=f"filter:{__name__}",
            model_provider="open-webui",
            session_id=body["metadata"]["chat_id"],
            ml_app=self.valves.ML_APP,
        )

        self.LLMObs.annotate(
            span=self.llm_span,
            input_data=get_last_user_message(body["messages"]),
        )

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        self.LLMObs.annotate(
            span=self.llm_span,
            output_data=get_last_assistant_message(body["messages"]),
        )

        self.llm_span.finish()
        self.LLMObs.flush()

        return body
