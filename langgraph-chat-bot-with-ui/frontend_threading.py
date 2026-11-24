import uuid

import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


# ************************************************************** Utility Functions ************************************************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(thread_id)
    st.session_state.messages = []


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def load_conversation(thread_id):
    return workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    ).values.get("messages", [])


# ************************************************************** Session Setup ************************************************************

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

add_thread(st.session_state["thread_id"])
CONFIG: RunnableConfig = {"configurable": {"thread_id": st.session_state["thread_id"]}}
# ---------------------------------------------------------------- Side Bar UI-------------------------------------------------------------------

st.sidebar.title("Chat Bot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state.chat_threads:
    if st.sidebar.button(thread_id):
        st.session_state.thread_id = thread_id
        message_history = load_conversation(thread_id)
        temp_messages = []
        # if message_history:
        for current_message in message_history:
            if isinstance(current_message, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({"role": role, "content": current_message.content})

        st.session_state.messages = temp_messages


# ----------------------------------------------------------------- Main UI ---------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])


chat_input = st.chat_input("Type your message here")

if chat_input:
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with st.chat_message("user"):
        st.text(chat_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content  # type: ignore
            for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=chat_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": ai_message})
