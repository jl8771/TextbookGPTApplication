import datasets
import os
import pickle
from langchain_core.documents import Document
from langchain.tools import Tool
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
#from langchain_community.retrievers import BM25Retriever

CORPUS_JSON_PATH = 'EngTextbooksPDF.json'
CORPUS_EMBEDDINGS_PATH = 'textbook_embeddings.pkl'
def get_documents() -> datasets.arrow_dataset.Dataset:
    textbook_dataset = datasets.load_dataset('json', data_files=CORPUS_JSON_PATH)['train']
    return textbook_dataset

def create_documents(textbook_dataset:datasets.arrow_dataset.Dataset) -> list[Document]:
    docs = [
        Document(
            page_content="\n".join([
                f"Title: {textbook['document']}",
                f"Page: {textbook['page']}",
                f"Text: {textbook['text']}"
            ]),
            metadata={"title": textbook["document"], "page": textbook["page"]}
        )
        for textbook in textbook_dataset
    ]
    return docs

def get_corpus_embeddings() -> Tensor:
    if not os.path.exists(CORPUS_EMBEDDINGS_PATH):
        textbook_dataset = get_documents()
        model = SentenceTransformer("msmarco-MiniLM-L12-cos-v5")
        document_embeddings = model.encode_document(textbook_dataset['text'], convert_to_tensor=True, show_progress_bar=True)
    else:
        with open(CORPUS_EMBEDDINGS_PATH, 'rb') as f:
            _, document_embeddings = pickle.load(f)
    return document_embeddings

def extract_text(query: str) -> str:
    """Retrieves detailed information about textbook pages"""
    model = SentenceTransformer("msmarco-MiniLM-L12-cos-v5")
    document_embeddings = get_corpus_embeddings()
    query_embedding = model.encode_query(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, document_embeddings, top_k=3)
    hits = hits[0]
    #TODO: Add re-ranking step here with cross-encoder to improve semantic search
    if hits:
        print("Extracted text from textbook with tool.")
        docs = create_documents(get_documents())
        return "\n\n".join([docs[hit['corpus_id']].page_content for hit in hits])
    else:
        print("Tried extracting text with tool, no text found.")
        return "No matching text information found."

textbook_info_tool = Tool(
    name="textbook_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about textbooks based on user queries."
)