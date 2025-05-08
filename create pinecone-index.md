# What is pinecone 
Pinecone is a vector database.
Instead of storing traditional data like numbers or text, it stores embeddings — numerical vectors that represent the meaning of sentences, paragraphs, or documents.

`Think of it like:`
> 📦 “A smart warehouse where you store and search the meaning of your content, not just the words.”   

## 🤖 Why do we use Pinecone in a RAG app?
In a RAG (Retrieval-Augmented Generation) app, you want the LLM (like Gemini or GPT-4) to answer questions based on your own data.

`So the process looks like this:`

1. 🧠 You turn your custom documents into embeddings using embedding model.

2. 📥 You store these embeddings in Pinecone
3. ❓ When the user asks a question, you convert it into an embedding
4. 🔍 Pinecone retrieves the most relevant content (based on vector similarity)
5. 💬 You send this content to Gemini/GPT to generate a contextual answer

<br>  

## Steps how to use pinecone

### ✅ Step 1: Create pinecone account
Click [here](https://docs.pinecone.io/guides/get-started/overview) and create you account on pinecone.

### ✅ Step 2: Get Your Pinecone API Key
In the Pinecone dashboard, go to API Keys create API Key then copy **API Key** and **Environment (e.g., gcp-starter, us-east1-gcp)**

🔐 Store this in your .env file or use st.secrets in Streamlit

``` toml
    PINECONE_API_KEY=your_key_here 
    PINECONE_ENV=your_env_here
```
Pinecone runs on the cloud — you access it over the internet using API Key   

### ✅ Step 3: Install Pinecone SDK
In your Python project, run:
``` 
      pip install pinecone-client
```   

### ✅ Step 4: Create the Index
Create the index in the pinecone by click on create index button and choose the custom configuration (e.g dimension 768) then click on create index button in the bottom.   

An index in Pinecone is like a folder or collection:

- It organizes your vector data
- You choose settings like dimension (e.g., 768 for Gemini embeddings)
- You can insert, update, delete, and search vectors in it   

### Step 5: In python project app

``` python 
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

# Set Gemini as the embedding model
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Load the pinecone API KEY
pinecone_client = Pinecone(api_key=st.secrets["Pinecone_API_KEY"])

# Connect to Pinecone index (must be created already in dashboard)
pinecone_index = pinecone_client.Index("index_name")

# Wrap the index with LlamaIndex's PineconeVectorStore
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create a storage context with that vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create a document and store it in the pinecone database
doc = Document(text="IELTS Speaking band 9 sample answers")
index = VectorStoreIndex.from_documents([doc], storage_context=storage_context)
```



**Sources**  

Youtube :  
https://www.youtube.com/watch?v=QUc2XvuJISM&list=PL3B-MqVuFtHTOdrR0jk45afPQLhostNaA&index=4&pp=iAQB

DataCamp:   
https://www.datacamp.com/tutorial/mastering-vector-databases-with-pinecone-tutorial 

Medium:  
https://medium.com/@jimwang3589/pinecone-a-specialized-vector-database-vectordb-3dd5c90cf77d#:~:text=Pinecone%20is%20a%20specialized%20Vector,large%20language%20models%20(LLMs).