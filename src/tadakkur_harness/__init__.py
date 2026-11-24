# pip install -qU langchain langchain-openrouter
from langchain.agents import create_agent
import asyncio

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}: 48 degree Celsius. 87% Humidity. 11 UV Index."

agent = create_agent(
    model="ollama:qwen3.5:2b",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

async def call_agent() -> None:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's the weather in Singapore?"}]}
    )
    print(result["messages"][-1].content)
    
def main() -> None:
    asyncio.run(call_agent())