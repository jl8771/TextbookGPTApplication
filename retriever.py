import datasets
import os
import pickle
from langchain_core.documents import Document
from langchain.tools import Tool
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from torch import Tensor
#from langchain_community.retrievers import BM25Retriever

CORPUS_JSON_PATH = 'EngTextbooksPDF_Cleaned.json'
CORPUS_EMBEDDINGS_PATH = 'textbook_embeddings.pkl'
def get_documents() -> datasets.arrow_dataset.Dataset:
    textbook_dataset = datasets.load_dataset('json', data_files=CORPUS_JSON_PATH)['train']
    return textbook_dataset

def create_documents(textbook_dataset:datasets.arrow_dataset.Dataset) -> list[Document]:
    """Creates a list of Document objects from the textbook dataset.

    Args:
        textbook_dataset (datasets.arrow_dataset.Dataset): The dataset containing textbook pages.

    Returns:
        list[Document]: A list of Document objects with page content and metadata.
    """
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
    """Retrieves or computes the corpus embeddings for the textbook dataset.

    Returns:
        Tensor: The tensor containing the document embeddings.
    """
    if not os.path.exists(CORPUS_EMBEDDINGS_PATH):
        textbook_dataset = get_documents()
        model = SentenceTransformer("msmarco-MiniLM-L12-cos-v5")
        document_embeddings = model.encode_document(textbook_dataset['text'], convert_to_tensor=True, show_progress_bar=True)
        with open(CORPUS_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(document_embeddings, f)
    else:
        with open(CORPUS_EMBEDDINGS_PATH, 'rb') as f:
            document_embeddings = pickle.load(f)
    return document_embeddings

def extract_text(query: str) -> str:
    """Extracts relevant textbook information based on the user's query using semantic search and reranking.

    Args:
        query (str): The user's query for textbook information.

    Returns:
        str: The extracted textbook pages as a string.
    """
    output_documents = []
    bi_encoder = SentenceTransformer("msmarco-MiniLM-L12-cos-v5")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    document_embeddings = get_corpus_embeddings()
    query_embedding = bi_encoder.encode_query(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, document_embeddings, top_k=32)
    hits = hits[0]
    if hits:
        print("Found text with semantic search. Proceeding to reranking...")
        pass
    else:
        print("Tried extracting text with tool, no text found.")
        return "No matching text information found."
    docs = create_documents(get_documents())
    cross_inputs = [[query, docs[hit['corpus_id']].page_content] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inputs)
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]
    ce_hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in ce_hits[:3]:
        output_documents.append(docs[hit['corpus_id']].page_content)
    return "\n\n".join(output_documents)

# Create the tool for LangChain
textbook_info_tool = Tool(
    name="textbook_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about textbooks based on user queries."
)