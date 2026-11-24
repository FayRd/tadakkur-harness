from langchain.agents import create_agent
from llm.list_of_llms import models
from tools.get_weather import get_weather
from prompts.assistant_prompts import prompts
import asyncio

### TODO: CREATE CONVERSATION CLASS

agent = create_agent(
    model=models["qwen3.5"]["ollama"],
    tools=[get_weather],
    system_prompt=prompts["purpose"],
)

messages = [{"role": "user", "content": "What's the weather in Singapore?"}]

async def call_agent(messages: list) -> str:
    result = await agent.ainvoke(
        {"messages": messages}
    )
    return result["messages"][-1].content

if __name__ == '__main__':
    print(asyncio.run(call_agent(messages=messages)))