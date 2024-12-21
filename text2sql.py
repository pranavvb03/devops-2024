import streamlit as st
import pandas as pd
import re
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
if 'table_info' not in st.session_state:
    st.session_state.table_info = None

# Set your Google API key
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

def create_db_from_csv(csv_file) -> str:
    """Create SQLite database from uploaded CSV file"""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_db.name
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Clean column names: remove spaces and special characters
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    # Create SQLite connection
    conn = sqlite3.connect(db_path)
    
    # Save DataFrame to SQLite
    table_name = 'data_table'
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    conn.close()
    return db_path

def get_table_info(db_path: str) -> str:
    """Get comprehensive table information"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("PRAGMA table_info(data_table);")
    columns = cursor.fetchall()
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM data_table;")
    row_count = cursor.fetchone()[0]
    
    # Get column information and sample data
    column_info = []
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM data_table;")
        distinct_count = cursor.fetchone()[0]
        cursor.execute(f"SELECT {col_name} FROM data_table LIMIT 3;")
        samples = [str(row[0]) for row in cursor.fetchall()]
        column_info.append(f"Column '{col_name}' (Type: {col_type}, Distinct Values: {distinct_count}, Examples: {', '.join(samples)})")
    
    conn.close()
    
    return f"""
    Table Name: data_table
    Total Rows: {row_count}
    
    Schema Information:
    {'\n'.join(column_info)}
    """

def clean_sql_query(query: str) -> str:
    """Clean and extract SQL query from the model's response"""
    # Remove markdown formatting
    query = re.sub(r'```sql|```', '', query)
    # Remove any trailing semicolons and extra whitespace
    query = query.strip().rstrip(';')
    # Add semicolon back for consistency
    return query + ';'

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

def create_vector_store(text: str, embeddings) -> FAISS:
    """Create FAISS vector store from text"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(text)
    return FAISS.from_texts(texts, embeddings)

# Create Streamlit interface
st.title("Interactive SQL Chatbot with RAG")
st.write("Upload a CSV file to create a database and ask questions in natural language!")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and st.session_state.db_path is None:
    # Create database from uploaded file
    st.session_state.db_path = create_db_from_csv(uploaded_file)
    
    # Get comprehensive table information
    st.session_state.table_info = get_table_info(st.session_state.db_path)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vector_store = create_vector_store(st.session_state.table_info, embeddings)
    
    st.success("Database created successfully!")

if st.session_state.db_path:
    # Initialize chat model
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Create prompt template
    prompt_template = """
    You are a SQL expert. Generate a SQL query based on the following database information and question.
    Return ONLY the SQL query without any explanations or decorations.
    If you cannot generate a valid query, respond with "I cannot answer this question with the available data."
    
    Database Information:
    {context}
    
    User Question: {question}
    
    Response:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    # Create chain
    chain = LLMChain(llm=chat_model, prompt=prompt)
    
    # Chat interface
    if question := st.chat_input("Ask any question about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        try:
            # Get relevant context from vector store
            docs = st.session_state.vector_store.similarity_search(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Generate SQL query
            sql_response = chain.run(context=context, question=question)
            
            # Check if the response indicates inability to answer
            if "cannot answer" in sql_response.lower():
                st.session_state.messages.append({"role": "assistant", "content": sql_response})
            else:
                # Clean and execute the query
                cleaned_query = clean_sql_query(sql_response)
                results = execute_sql_query(st.session_state.db_path, cleaned_query)
                
                # Format response
                response = f"SQL Query:\n```sql\n{cleaned_query}\n```\n\nResults:\n{results.to_markdown()}"
                st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"Error: {str(e)}\nPlease try rephrasing your question."
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
atexit.register(cleanup)
