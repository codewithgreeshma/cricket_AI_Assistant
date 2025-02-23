import os
import logging
import pandas as pd
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "data")
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH")

# Load & Process Multiple CSV Files using LangChain CSVLoader
# def load_data_from_directory(directory_path):
#     all_documents = []
#     index_creator = VectorstoreIndexCreator()
#     # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
#
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".csv"):
#             file_path = os.path.join(directory_path, filename)
#             base_name = os.path.splitext(os.path.basename(filename))[0]  # Get filename without extension
#             csv_path = os.path.join(directory_path, f"{base_name}.csv")
#
#             logger.info(f"Processing file: {filename}")
#             loader = CSVLoader(file_path)
#             documents = loader.load()
#             for doc in documents:
#                 doc.metadata["source"] = base_name  # Store filename without extension in metadata
#             all_documents.extend(documents)
#     logger.info("Splitting documents for better retrieval...")
#     split_documents = text_splitter.split_documents(all_documents)
#
#     logger.info("Creating vector store with FAISS...")
#     vector_store = FAISS.from_documents(split_documents, embedding_model)
#     logger.info("Vector store created successfully.")
#     # logger.info("Creating vector store...")
#     # vector_store = index_creator.from_documents(all_documents)
#     # logger.info("Vector store created successfully.")
#     return vector_store
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def load_data_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            loader = CSVLoader(file_path=file_path)
            documents.extend(loader.load())  # Load CSV as documents
    return documents
DATA_DIR = 'data/cricket_data'
if os.path.exists(FAISS_DB_PATH) and os.path.exists(FAISS_METADATA_PATH):
    logger.info("Loading existing FAISS vector store...")

    # Load FAISS index
    vector_store = FAISS.load_local(FAISS_DB_PATH, embedding_model)

    # Load document metadata
    with open(FAISS_METADATA_PATH, "rb") as f:
        split_documents = pickle.load(f)

    logger.info("FAISS vector store loaded successfully.")
else:
    logger.info("No existing vector store found. Creating new one...")
    # Convert CSVs into a text format for embedding
    documents = load_data_from_directory(DATA_DIR)

    split_documents = text_splitter.split_documents(documents)

    logger.info("Creating vector store with FAISS...")
    vector_store = FAISS.from_documents(split_documents, embedding_model)
    logger.info("Vector store created successfully.")
    # Save the FAISS index for future runs
    os.makedirs("vector_store", exist_ok=True)
    vector_store.save_local(FAISS_DB_PATH)

    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(split_documents, f)
    logger.info("New FAISS vector store saved.")

# Initialize API
app = FastAPI()

logger.info("Loading data from directory...")
data_vector_store = vector_store
retriever = data_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Improved retrieval

# Load Meta LLM
# llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_type="llama", config={"context_length": 4096})
# llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_type="mistral")


llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_type="mistral",
                    config={"max_new_tokens": 512, "context_length": 2048})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# qa_chain = RetrievalQA(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=data_vector_store.as_retriever())
# logger.info("Data loading completed.")


# Pydantic Models
class PlayerStatsRequest(BaseModel):
    name: str


class TeamComparisonRequest(BaseModel):
    team1: str
    team2: str


class MatchSummaryRequest(BaseModel):
    match_id: int


class ChatRequest(BaseModel):
    query: str


@app.post("/player_stats")
def get_player_stats(request: PlayerStatsRequest):
    logger.info(f"Received player stats request for {request.name}")
    return qa_chain.run(f"Show career statistics for the player{request.name}")


@app.post("/team_comparison")
def team_comparison(request: TeamComparisonRequest):
    logger.info(f"Received team comparison request: {request.team1} vs {request.team2}")
    return qa_chain.run(f"Compare {request.team1} and {request.team2} cricket teams and return records")


@app.post("/match_summary")
def match_summary(request: MatchSummaryRequest):
    logger.info(f"Received match summary request for match ID {request.match_id}")
    return qa_chain.run(f"Summarize the match with ID {request.match_id}")


@app.post("/chat")
def chat_with_ai(request: ChatRequest):
    logger.info(f"Received chat query: {request.query}")
    return qa_chain.run(request.query)


if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
