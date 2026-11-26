from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools.get_weather import get_weather
from prompts.assistant_prompts import prompts
from configs.config import OLLAMA_MODEL
from datetime import datetime, timezone
from typing import Any
import asyncio

### TODO: CREATE CONVERSATION CLASS
class Conversation:
    def __init__(self, id: str, name: str, agent: Any, messages: list = []):
        self.id = id
        self.name = name
        self.agent = agent
        self.messages = messages
        
    def get_details(self):
        return f"|{'ID':^15}|{'Name':^20}|\n|{self.id:^15}|{self.name:^20}|"
    
    async def _get_chat_response(self, message: str) -> str:
        try:
            response = await agent.ainvoke(
                input={"messages": message},
                config={"configurable": {"thread_id": self.id}}
            )
        except Exception as e:
            return {
                "type": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": f"Error while getting response: {e}"
                }
        return response
    
    async def send_message(self, message: str):
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
            })
        response = await self._get_chat_response({
            "role": "user",
            "content": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
            })
        self.messages.append({
            "role": "assistant",
            "content": response["messages"][-1].content,
            "timestamp": datetime.now(timezone.utc).isoformat()
            })
        return self.messages[-1]
    
    def _format_chat_history(self, messages):
        formatted_messages: str = ""
        for index, message in enumerate(messages):
            formatted_messages += f"{index+1: <3}. {message["role"]: <10}| {message["content"]: <10}\n"
        return formatted_messages
        
    def get_chat_history(self):
        return self._format_chat_history(self.messages)
        
checkpointer = InMemorySaver()
agent = create_agent(
    model=OLLAMA_MODEL,
    tools=[get_weather],
    system_prompt=prompts["purpose"],
    checkpointer=checkpointer
)

if __name__ == '__main__':
    asyncio.set_event_loop(asyncio.new_event_loop())
    prompt = input("> ")
    conversation_instance = Conversation(id="1", name="Untitled", agent=agent)
    while prompt != "/bye":
        if prompt == "/history":
            print(f"<hist>\n{conversation_instance.get_chat_history()}\n</hist>")
        elif prompt == "/info":
            print(f"<info>\n{conversation_instance.get_details()}\n</info>")
        else:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(conversation_instance.send_message(prompt))
            print(f">> {response["content"]}")
        prompt = input("> ")
    print(">> Bye!")