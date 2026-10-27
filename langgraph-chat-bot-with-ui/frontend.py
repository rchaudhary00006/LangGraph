import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}

st.title("A Simple Chat Bot with UI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])


chat_input = st.chat_input("Type your message here")

if chat_input:
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with st.chat_message("user"):
        st.text(chat_input)

    result = workflow.invoke(
        {"messages": [HumanMessage(content=chat_input)]}, config=config
    )

    ai_message = result["messages"][-1].content

    st.session_state.messages.append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)
