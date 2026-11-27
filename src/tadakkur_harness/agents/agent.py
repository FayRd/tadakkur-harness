from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools.get_weather import get_weather
from prompts.assistant_prompts import prompts
from configs.config import OLLAMA_MODEL
from datetime import datetime, timezone
from typing import Any
import asyncio

class Agent:
    """Create an agent.
    """
    def __init__(self, model: str, tools: list | None, system_prompt: str, middleware: list, debug: bool = False):
        self.graph = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            debug=debug
        )
    async def test_agent(self):
        """Sends a test prompt and prints the raw response.
        """
        response = await self.graph.ainvoke("Hello, this is a test prompt!")
        print(response)
        
class Conversation:
    """Basic interaction with the agent in a conversation-like structure.
    """
    def __init__(self, id: str, name: str, agent: Any, messages: list = []):
        self.id = id
        self.name = name
        self.agent = agent
        self.messages = messages
        
    def get_details(self):
        """Retrieves the conversation ID and name.

        Returns:
            str: A single-row table of this conversation's details
        """
        return f"|{'ID':^15}|{'Name':^20}|\n|{self.id:^15}|{self.name:^20}|"
    
    async def _get_chat_response(self, message: str) -> dict:
        """Invokes the agent with the raw response or error message.

        Args:
            message (str): The user's prompt/message to be sent.

        Returns:
            dict: The raw response or exception payload.
        """
        try:
            response = await self.agent.ainvoke(
                input={"messages": message}
            )
        except Exception as e:
            return {
                "type": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": f"Error while getting response: {e}"
                }
        return response
    
    async def send_message(self, message: str) -> dict:
        """Inserts this turn's messages to the chat history and retrieve agent response.

        Args:
            message (str): The user's prompt/message to be sent.

        Returns:
            dict: The assistant's final response in this turn.
        """
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
    
    def _format_chat_history(self, messages: list) -> str:
        """Formats the raw chat history in more readable format.

        Args:
            messages (list): The chat history for this conversation.

        Returns:
            str: The formatted chat history.
        """
        formatted_messages: str = ""
        for index, message in enumerate(messages):
            formatted_messages += f"\n{index+1}. [{message["role"]}]\n  {message["content"]: <10}\n"
        return formatted_messages
        
    def get_chat_history(self) -> str:
        """Retrieves this conversation's chat history.

        Returns:
            str: The chat chistory.
        """
        return self._format_chat_history(self.messages)
        

if __name__ == '__main__':
    asyncio.set_event_loop(asyncio.new_event_loop())
    agent_instance = Agent(OLLAMA_MODEL, [get_weather], prompts["purpose"], middleware=None, debug=True)
    agent = agent_instance.graph
    prompt = input("> ")
    conversation_instance = Conversation(id="1", name="Untitled", agent=agent)
    while prompt != "/bye":
        if prompt[0] == "/":      
            if prompt == "/history":
                print(f"<hist>\n{conversation_instance.get_chat_history()}\n</hist>")
            elif prompt == "/info":
                print(f"<info>\n{conversation_instance.get_details()}\n</info>")
            else:
                print(f"Unknown command.")
        else:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(conversation_instance.send_message(prompt))
            print(f">> {response["content"]}")
        prompt = input("> ")
    print(">> Bye!")