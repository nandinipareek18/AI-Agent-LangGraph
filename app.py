import streamlit as st

from ai_agent import get_response_from_ai_agent


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("🤖 AI Agent")
st.write("Powered by Groq + LangGraph + Tavily")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Settings")


# Only Groq
llm_id = st.sidebar.text_input(
    "Groq Model",
    value="openai/gpt-oss-20b"
)


allow_search = st.sidebar.checkbox(
    "Enable Web Search",
    value=False
)


system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a helpful AI assistant."
)


# =====================================================
# USER INPUT
# =====================================================

query = st.text_input(
    "Enter your question:"
)


# =====================================================
# GENERATE RESPONSE
# =====================================================

if st.button("Generate Response"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            try:

                response = get_response_from_ai_agent(
                    llm_id=llm_id,
                    query=[query],
                    allow_search=allow_search,
                    system_prompt=system_prompt
                )

                st.subheader("🤖 AI Response")
                st.write(response)

            except Exception as e:

                st.error(f"Error: {e}")