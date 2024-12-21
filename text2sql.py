import streamlit as st
import pandas as pd
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.messages  import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from typing import List
import tempfile

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'db_path' not in st.session_state:
    st.session_state.db_path = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Set your Google API key
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

def create_db_from_csv(csv_file) -> str:
    """Create SQLite database from uploaded CSV file"""
    # Create a temporary file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_db.name
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create SQLite connection
    conn = sqlite3.connect(db_path)
    
    # Save DataFrame to SQLite
    table_name = 'data_table'
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    conn.close()
    return db_path

def get_table_schema(db_path: str) -> str:
    """Get schema information from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schema = cursor.fetchall()
    
    conn.close()
    return '\n'.join([s[0] for s in schema if s[0] is not None])

def create_vector_store(schema: str, embeddings):
    """Create FAISS vector store from schema information"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(schema)
    
    return FAISS.from_texts(texts, embeddings)

def execute_sql_query(db_path: str, query: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        raise e

# Create Streamlit interface
st.title("Interactive SQL Chatbot with RAG")
st.write("Upload a CSV file to create a database and ask questions in natural language!")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and st.session_state.db_path is None:
    # Create database from uploaded file
    st.session_state.db_path = create_db_from_csv(uploaded_file)
    
    # Get schema information
    schema = get_table_schema(st.session_state.db_path)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vector_store = create_vector_store(schema, embeddings)
    
    st.success("Database created successfully!")

if st.session_state.db_path:
    # Initialize chat model
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    # Create prompt template
    prompt_template = """
    You are a SQL expert. Given the following question and database schema context, generate a MySQL-compatible SQL query.
    Only return the SQL query without any explanations.
    
    Context: {context}
    Question: {question}
    
    SQL Query:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    # Create chain
    chain = LLMChain(llm=chat_model, prompt=prompt)
    
    # Chat interface
    if question := st.chat_input("Ask a question about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get relevant context from vector store
        docs = st.session_state.vector_store.similarity_search(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        try:
            # Generate SQL query
            sql_query = chain.run(context=context, question=question)
            
            # Execute query
            results = execute_sql_query(st.session_state.db_path, sql_query)
            
            # Add response to chat history
            response = f"SQL Query:\n```sql\n{sql_query}\n```\n\nResults:\n{results.to_markdown()}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Cleanup on session end
def cleanup():
    if st.session_state.db_path and os.path.exists(st.session_state.db_path):
        os.unlink(st.session_state.db_path)

# Register cleanup
import atexit
atexit.register(cleanup)
