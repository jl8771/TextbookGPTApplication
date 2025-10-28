from agent import AgentConfig
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langfuse.langchain import CallbackHandler

class ChatSession:
    """Manages a chat session with the textbook assistant agent.
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
        
        system_prompt (str): The system prompt to initialize the assistant's behavior. Default is `None`
        to use a predefined prompt to specialize in retrieving and summarizing information from academic textbooks.
    """
    def __init__(self, model_provider: str = "qwen", system_prompt: str = None):
        os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
        os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
        
        self.provider = model_provider
        self.config = AgentConfig(provider=self.provider)
        self.assistant = self.config.get_agent()
        self.model_name = self.config.get_model_fullname()
        self.messages = []
        self.assistant_messages = []
        self.langfuse_handler = CallbackHandler()
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = """You are a helpful and knowledgeable assistant specializing in 
        retrieving and summarizing information from academic textbooks.
        When possible, you should use the retrieval tool to find relevant
        passages from textbooks to support your answers. Provide the name
        and page numbers you used in your answers. If you are unsure about
        an answer, it is better to say "I don't know" than to provide
        incorrect information.
        
        Always use the tools when the user asks for specific information,
        especially the retrieval tool for textbook content. Always use this
        tool even in long conversations to ensure accuracy. Use the retrieval
        tool whenever the user asks about specific facts, concepts, or
        "what is ..." questions or "define ..." questions.
        
        Whenever you have a block of text that should be rendered as LaTeX,
        format it using dollar signs for inline math and double dollar signs
        for display math. For example, use $...$ for inline math and $$...$$
        for display math.
        """
        print(f"Chat session with assistant initialized using model: {self.model_name}")
        
    def get_provider(self) -> str:
        return self.model_name

    def start_chat(self, query: str) -> str:
        """Starts a new chat session with the assistant.

        Args:
            query (str): The user's initial query to start the chat.

        Returns:
            str: The assistant's response to the initial query.
        """
        if len(self.messages) == 0:
            response = self.assistant.invoke(
                input={"messages": [SystemMessage(content=self.system_prompt)] + [HumanMessage(content=query)]},
                config={"callbacks": [self.langfuse_handler]}
            )
            self.assistant_messages = response
            self.messages.append({"User": query})
            self.messages.append({"Assistant": response['messages'][-1].content})
        else:
            print("Error: Chat session already started.")
            return "Error: Chat session already started. Clear chat to start a new session."
        return response['messages'][-1].content
            
    def continue_chat(self, query: str) -> str:
        """Continues an existing chat session with the assistant.

        Args:
            query (str): The user's follow on query to continue the chat.

        Returns:
            str: The assistant's response to the follow on query.
        """
        if len(self.messages) != 0:
            response = self.assistant.invoke(
                input={"messages": self.assistant_messages['messages'] + [HumanMessage(content=query)]},
                config={"callbacks": [self.langfuse_handler]}
            )
            self.assistant_messages = response
            self.messages.append({"User": query})
            self.messages.append({"Assistant": response['messages'][-1].content})
        else:
            print("Error: Chat session not started yet. Please start a chat first.")
            return "Error: Chat session not started yet. Please start a chat first."
        return response['messages'][-1].content
    
    def clear_chat(self) -> None:
        """Clears the current chat session, resetting messages and assistant state.
        """
        self.messages = []
        self.assistant_messages = []
        print("Chat session cleared.")