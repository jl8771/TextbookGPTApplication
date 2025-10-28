from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

from tools import search_tool, check_blank_page_tool, summarize_page_tool
from retriever import textbook_info_tool
from model_provider import get_model_provider, get_model_fullname

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class AgentConfig:
    """Configuration class for setting up the textbook assistant agent with different LLM providers and tools.
    Includes support for:
    - OpenAI GPT models
    - Alibaba Qwen models
    - Meta Llama models
    - MistralAI Mistral models
    - DeepseekAI DeepSeek-R1
    - NVIDIA OpenReasoning-Nemotron-32B
    - MoonshotAI Kimi-K2-Instruct-0905
    
    Attributes:
        provider (str): The name of the LLM provider to use. Default is `qwen`. Available providers:
        `openai`, `qwen`, `llama`, `mistral`, `deepseek`, `nemotron`, `kimi`
    """
    def __init__(self, provider: str = "qwen"):
        self.provider = provider
        self.llm = self.set_provider(provider)
        
    def get_model_fullname(self) -> str:
        """Retrieves the full name of the selected LLM model based on the provider.

        Returns:
            str: The full name of the LLM model.
        """
        return get_model_fullname(self.provider)

    def set_provider(self, provider_name: str = "qwen"):
        """Sets the LLM provider based on the specified name.

        Args:
            provider_name (str, optional): Name of the model provider. Defaults to "qwen".

        Returns:
            _type_: An instance of the selected LLM model.
        """
        print("Setting model provider to:", provider_name)
        available_providers = ["qwen", "openai", "llama", "mistral", "deepseek", "nemotron", "kimi"]
        if provider_name in available_providers:
            return get_model_provider(provider_name)

    def get_agent(self):
        """Configures and returns the textbook assistant agent with the selected LLM and tools.

        Returns:
            _type_: A configured StateGraph representing the textbook assistant agent.
        """
        llm = self.llm
        chat_tools = [textbook_info_tool, summarize_page_tool, search_tool]
        llm_with_tools = llm.bind_tools(chat_tools)

        def assistant(state: AgentState):
            return {
                "messages": [llm_with_tools.invoke(state["messages"])],
            }
        
        builder = StateGraph(AgentState)

        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools=chat_tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        memory = InMemorySaver() #TODO: Replace with database-based checkpointer
        textbook_assistant = builder.compile()
        return textbook_assistant