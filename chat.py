import streamlit as st

from dotenv import load_dotenv

from llm import get_ai_response

st.set_page_config(page_title="Cliamte Virtual Assitant", page_icon=":🌏", layout="wide")

st.title("🌏 Cliamte Virtual Assitant")
st.caption("Powered by State of the Climate - BAMS")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.w(message["content"])

if user_question := st.chat_input(placeholder="Ask me anything"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("Generating..."):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
