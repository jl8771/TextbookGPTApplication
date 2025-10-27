import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
#from langchain_anthropic import Anthropic
#from langchain_ollama import Ollama
#from langchain_mistralai import MistralAI
#from langchain_deepseek import Deepseek
#from langchain_perplexity import PerplexityAI
#from langchain_xai import XAI

available_models = [
    "openai",
    "qwen",
    "llama",
    "kimi",
    "mistral",
    "deepseek",
    "nemotron",
    #"claude", # TODO: Add support for Anthropic Claude
    #"xai", # TODO: Add support for xAI
    #"perplexity", # TODO: Add support for Perplexity AI
]

def get_model_provider(model_name: str):
    """Selects and returns the appropriate LLM model based on the specified provider name.

    Args:
        model_name (str): Name of the model provider. Options include:
            - `openai`: OpenAI GPT-4o
            - `qwen`: Alibaba Qwen3-Coder-30B-A3B-Instruct
            - `llama`: Meta Llama-4-Maverick-17B-128E-Instruct
            - `mistral`: Mistral-Small-3.2-24B-Instruct-2506
            - `deepseek`: DeepseekAI DeepSeek-R1
            - `nemotron`: NVIDIA OpenReasoning-Nemotron-32B
            - `kimi`: MoonshotAI Kimi-K2-Instruct-0905

    Raises:
        ValueError: If the specified model_name is not supported.

    Returns:
        _type_: An instance of the selected LLM model.
    """
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {available_models}")
    
    if model_name == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            #model="gpt-5-main",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    elif model_name == "qwen":
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            #repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        return ChatHuggingFace(llm=llm, verbose=True)
    elif model_name == "llama":
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            #repo_id="meta-llama/Llama-3.3-70B-Instruct"
            #repo_id="meta-llama/Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        return ChatHuggingFace(llm=llm, verbose=True)
    elif model_name == "mistral":
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            #repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        return ChatHuggingFace(llm=llm, verbose=True)
    elif model_name == "deepseek":
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        return ChatHuggingFace(llm=llm, verbose=True)
    elif model_name == "nemotron":
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id="nvidia/OpenReasoning-Nemotron-32B",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        return ChatHuggingFace(llm=llm, verbose=True)
    elif model_name == "kimi":
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm = HuggingFaceEndpoint(
            repo_id="moonshotai/Kimi-K2-Instruct-0905",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        return ChatHuggingFace(llm=llm, verbose=True)
        
def get_model_fullname(model_name: str):
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {available_models}")
    
    if model_name == "openai":
        return "OpenAI GPT-4o"
    elif model_name == "qwen":
        return "Alibaba Qwen3-Coder-30B-A3B-Instruct"
    elif model_name == "llama":
        return "Meta Llama-4-Maverick-17B-128E-Instruct"
    elif model_name == "mistral":
        return "MistralAI Mistral-Small-3.2-24B-Instruct-2506"
    elif model_name == "deepseek":
        return "DeepseekAI DeepSeek-R1"
    elif model_name == "nemotron":
        return "NVIDIA OpenReasoning-Nemotron-32B"
    elif model_name == "kimi":
        return "MoonshotAI Kimi-K2-Instruct-0905"    