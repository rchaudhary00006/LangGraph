from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")
conn = sqlite3.connect(
    database="chatbot.db", check_same_thread=False
)  # Create a connection to the SQLite database and that will be created under this current directory. If the database does not exist, it will be created automatically. The check_same_thread=False parameter allows the connection to be used across multiple threads.

# Checkpointer is responsible for saving and loading the state of the workflow. In this case, we are using a SQLite database to store the state of the workflow. The SqliteSaver class is used to create a checkpointer that saves the state of the workflow to a SQLite database. The conn parameter is used to pass the connection object to the SqliteSaver class.

checkpointer = SqliteSaver(conn=conn)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph = StateGraph(ChatState)


def chatNode(state: ChatState):

    prompt = [SystemMessage(content="You are a helpful assistant")] + state["messages"]

    response = llm.invoke(prompt)

    return {"messages": response}


graph.add_node("chat_node", chatNode)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

# ai_response = workflow.invoke(
#     {"messages": [HumanMessage(content="What is my name?")]},
#     config={"configurable": {"thread_id": "thread-1"}},
# )

# # print(ai_response)

# workflow_state = workflow.get_state(config={"configurable": {"thread_id": "thread-1"}})

# print(workflow_state.values.get("messages", []))


# checkpointer.list(None)  # List all the checkpoints in the database. The None parameter is used to list all the checkpoints in the database. The list method returns a list of all the checkpoints in the database. Each checkpoint is represented as a dictionary containing the configuration and values of the checkpoint.
def retrive_all_threads():
    all_unique_threads = set()
    for checkpoint in checkpointer.list(None):
        all_unique_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_unique_threads)
