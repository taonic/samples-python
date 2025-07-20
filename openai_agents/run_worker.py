from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from agents import (
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    set_tracing_disabled,
)

from temporalio.client import Client
from temporalio.contrib.openai_agents import (
    ModelActivity,
    ModelActivityParameters,
    set_open_ai_agent_temporal_overrides,
)
from openai import AsyncOpenAI
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from openai_agents.workflows.agents_as_tools_workflow import AgentsAsToolsWorkflow
from openai_agents.workflows.customer_service_workflow import CustomerServiceWorkflow
from openai_agents.workflows.get_weather_activity import get_weather
from openai_agents.workflows.hello_world_workflow import HelloWorldAgent
from openai_agents.workflows.research_bot_workflow import ResearchWorkflow
from openai_agents.workflows.tools_workflow import ToolsWorkflow

BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)


CUSTOM_MODEL_PROVIDER = CustomModelProvider()

async def main():
    with set_open_ai_agent_temporal_overrides(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=60),
        ),
    ):
        # Create client connected to server at the given address
        client = await Client.connect(
            "localhost:7233",
            data_converter=pydantic_data_converter,
        )

        worker = Worker(
            client,
            task_queue="openai-agents-task-queue",
            workflows=[
                HelloWorldAgent,
                ToolsWorkflow,
                ResearchWorkflow,
                CustomerServiceWorkflow,
                AgentsAsToolsWorkflow,
            ],
            activities=[
                ModelActivity(model_provider=CUSTOM_MODEL_PROVIDER).invoke_model_activity,
                get_weather,
            ],
        )
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
