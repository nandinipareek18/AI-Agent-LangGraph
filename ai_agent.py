import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =====================================================
# AI AGENT
# =====================================================

def get_response_from_ai_agent(
    llm_id: str,
    query: list,
    allow_search: bool,
    system_prompt: str
):

    # -------------------------------------------------
    # GROQ LLM
    # -------------------------------------------------

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    llm = ChatGroq(
        model=llm_id,
        api_key=GROQ_API_KEY
    )

    # -------------------------------------------------
    # TAVILY SEARCH TOOL
    # -------------------------------------------------

    tools = []

    if allow_search:

        if not TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY is missing. Check your .env file."
            )

        tools.append(
            TavilySearchResults(
                max_results=2,
                tavily_api_key=TAVILY_API_KEY
            )
        )

    # -------------------------------------------------
    # LANGGRAPH AGENT
    # -------------------------------------------------

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # -------------------------------------------------
    # STATE
    # -------------------------------------------------

    state = {
        "messages": [
            ("system", system_prompt),
            ("user", query[0])
        ]
    }

    # -------------------------------------------------
    # RUN AGENT
    # -------------------------------------------------

    response = agent.invoke(state)

    # -------------------------------------------------
    # GET FINAL RESPONSE
    # -------------------------------------------------

    messages = response.get("messages", [])

    ai_messages = [
        message.content
        for message in messages
        if isinstance(message, AIMessage)
    ]

    if not ai_messages:
        return "No response generated."

    return ai_messages[-1]