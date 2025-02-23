"""
Cricket Intelligence Assistant (CIA)

2. Large Language Model (LLM) Integration
3.API Deployment & Chatbot Integration

"""

import os
import logging
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

# get constants from environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR")  # directory where CSV data files are stored.
FAISS_DB_PATH = os.getenv(
    "FAISS_DB_PATH")  # Path to store/load FAISS vector database (used for storing document embeddings)
FAISS_METADATA_PATH = os.getenv(
    "FAISS_METADATA_PATH")  # Path to store/load metadata related to FAISS (e.g., document details for retrieval)

"""
Split large text into smaller chunks for embedding.
This helps in better retrieval and efficient vector search.
- chunk_size = 256: Each chunk contains up to 256 characters.
- chunk_overlap = 50: Ensures 50 characters from the previous chunk overlap with the next one.
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)

"""
To load a pre-trained sentence embedding model for converting text into numerical vectors.
The selected model is `all-MiniLM-L6-v2`, a lightweight transformer model optimized for embeddings.
"""
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Function to load all CSV files from a given directory and extract their contents.
def load_data_from_directory(directory):
    """
        Loads all CSV files from the specified directory and returns their contents as a list of documents.

        Args:
            directory (str): Path to the folder containing CSV files.

        Returns:
            list: A list of document objects containing the text extracted from each CSV file.
    """
    documents = []  # Initialize an empty list to store extracted document data
    for filename in os.listdir(directory):  # Iterate over all files in the directory
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            # Load CSV file and convert it into text-based document format
            loader = CSVLoader(file_path=file_path)
            documents.extend(loader.load())  # Append loaded documents to the list
    return documents


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

    # Split the loaded text data into smaller chunks
    # This is necessary to create meaningful vector representations for retrieval.
    split_documents = text_splitter.split_documents(documents)

    logger.info("Creating vector store with FAISS...")

    # Create a new FAISS vector store using document embeddings
    # FAISS will store the vectorized representations of our text data.
    vector_store = FAISS.from_documents(split_documents, embedding_model)
    logger.info("Vector store created successfully.")
    # Save the FAISS index for future runs
    os.makedirs("vector_store", exist_ok=True)
    vector_store.save_local(FAISS_DB_PATH)

    # Save the document metadata (text chunks) for future retrieval
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(split_documents, f)  # Serialize and store the document chunks
    logger.info("New FAISS vector store saved.")

# Initialize API
app = FastAPI()

logger.info("Loading data from directory...")
data_vector_store = vector_store

# Create a retriever from the FAISS vector store
# performs similarity search when retrieving relevant text chunks
# `search_type="similarity"` ensures relevant text chunks are retrieved based on semantic similarity
# `k=5` means that the top 5 most relevant text chunks will be fetched for answering queries
retriever = data_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Improved retrieval

# Load Meta LLM
# llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", model_type="llama", config={"context_length": 4096})
# llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_type="mistral")

# Load the Mistral-7B-Instruct model with specific configuration settings
# - `"max_new_tokens": 512` limits response length to 512 tokens to control output size
# - `"context_length": 2048` sets the maximum number of tokens that the model can process at once
llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_type="mistral",
                    config={"max_new_tokens": 512, "context_length": 2048})

# Create a retrieval-augmented generation (RAG) pipeline
# - Uses the retriever to fetch relevant text chunks from the FAISS database
# - Passes the retrieved text to the LLM for generating responses
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# qa_chain = RetrievalQA(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=data_vector_store.as_retriever())
# logger.info("Data loading completed.")


# ================================
# Pydantic Models for API Requests
# ================================
# Define request model for fetching player statistics
class PlayerStatsRequest(BaseModel):
    name: str  # The player's name as a string input


# Define request model for team comparison
class TeamComparisonRequest(BaseModel):
    team1: str
    team2: str


# Define request model for fetching match summary
class MatchSummaryRequest(BaseModel):
    match_id: int  # Unique match identifier as an integer


# Define request model for chat-based interactions
class ChatRequest(BaseModel):
    query: str


# ================================
# FastAPI Endpoints
# ================================
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


# ================================
# Start FastAPI Server
# ================================
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    import uvicorn

    # Run the FastAPI application on `0.0.0.0:8000`
    uvicorn.run(app, host="0.0.0.0", port=8000)
