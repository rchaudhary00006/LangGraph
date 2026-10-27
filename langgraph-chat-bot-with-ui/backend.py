from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
checkpointer = InMemorySaver()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph = StateGraph(ChatState)


def chat_node(state: ChatState):
    prompt = (
        f"{SystemMessage(content='You are an helpful assistant')} + {state['messages']}"
    )

    response = llm.invoke(prompt)

    return {"messages": response}


graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

# for message_chunk, metadata in workflow.stream(
#     {"messages": [HumanMessage(content="What is recepie to make pasta")]},
#     config={"configurable": {"thread_id": "thread-1"}},
#     stream_mode="messages",
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)
