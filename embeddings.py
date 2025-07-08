
# Import Libraries

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from pinecone import Pinecone 
import fitz   # PyMuPDF for reading PDFs
import os 
import streamlit as st 
import re


# Load API KEY 
# Fetch GEMINI API either from .env or from st.secrets  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])        

# Configure Gemini Embeddings
# Explicitly call the Gemini Embeddings because by default llama use openAI
embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001",
    api_key=os.environ["GEMINI_API_KEY"]
)


# For Text Generation
Settings.llm = GoogleGenAI(
    model = "gemini-1.5-flash-latest",
    api_key=os.environ["GEMINI_API_KEY"]
)


# Load the pinecone API KEY
pinecone_client = Pinecone(api_key=st.secrets["Pinecone_API_KEY"])

# Connect to Pinecone index (must be created already in dashboard)
pinecone_index = pinecone_client.Index("ielts-assistant-index")

# Initialize vector store (e.g., Pinecone)
# Step 2: Create a storage context with that vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create a storage context with that vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = f"Document: {os.path.basename(pdf_path)}\n\n"
    text += "\n".join(page.get_text("text") for page in doc)
    return text    


# # Checks whether a specific document is already stored in your Pinecone index
# def already_indexed(doc_id):
#     try:
#         # use query instead of fetch 
#         # Searches vectors by using metadata filters
#         query_result = pinecone_index.query(
#             vector=[0.0] * 768,  # dummy vector
#             top_k=1,
#             filter={"doc_id": {"$eq": doc_id}}
#         )
#         return len(query_result.matches) > 0
#     except Exception as e:
#         print(f"Error checking Pinecone for {doc_id}: {e}")
#         return False



# # Load and Index PDFs with Pinecone
# def load_data():
#     data_dir = "./data"
    
#     # Check directory exit or not
#     if not os.path.exists(data_dir):
#         st.warning(
#             f"Data directory '{data_dir}' not found."
#         )
#         return None
    
    
#     # Check pdf file exist in the directory or not
#     pdf_files = [file for file in os.listdir(data_dir) if file.endswith(".pdf")]
#     if not pdf_files:
#         st.warning(
#             "No PDF files found in the data directory."
#         )
#         return None
    
    
#     # Split into manageable chunks (~512 tokens each)
#     text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

#     with st.spinner("Loading and indexing PDFs into Pinecone..."):
#       docs = []
#       for filename in pdf_files:
#         # use filename as unique ID in Pinecone
#         doc_id = filename  
        
#         if already_indexed(doc_id):
#             print(f"Skipping already indexed: {filename}")
#             continue
        
        
#         pdf_files = os.path.join(data_dir, filename)
#         text = extract_text_from_pdf(pdf_files)
#        # docs.append(Document(text = text, id = doc_id))  #a attach doc_id for tracking
#         node = text_splitter.get_nodes_from_documents([TextNode(text=text, metadata={"doc_id":doc_id})])  # Assign doc_id as vector ID
#         docs.extend(node)

    
#        # Build index and push to Pinecone
#     if docs:
#         index = VectorStoreIndex(docs, embed_model=embed_model, storage_context=storage_context)
#         return index



# # Load and store into Pinecone
# index = load_data()


# Semantic Search, 
def genarate_response(prompt):
    
    # Get Previous chat history from langchain memory  
    chat_history = st.session_state.langchain_memory.load_memory_variables({})
    
    # Extract text from history 
    chat_history_text = chat_history.get("chat_history_text", "")
    
    
    # Combine chat-history-text + prompt
    combined_prompt = f"{chat_history_text}\nUser: {prompt}"
    
    # Connect chatbot to the saved knowledge (PDFs, text, etc.) stored in a vector database — Pinecone
    index = VectorStoreIndex.from_vector_store(embed_model=embed_model,vector_store=vector_store )
    
    # Tells the app how to search and how many results to fetch
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    response = query_engine.query(combined_prompt)
    response = response.response
    
    sentences = re.split(r'(?<=[.!?]) +', response)
    
    for sentence in sentences:
        yield sentence.strip()
