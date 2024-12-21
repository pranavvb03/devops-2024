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
    
    # Create detailed schema information
    schema_info = []
    for col in columns:
        name, dtype = col[1], col[2]
        cursor.execute(f"SELECT MIN({name}), MAX({name}) FROM data_table;")
        min_val, max_val = cursor.fetchone()
        schema_info.append(f"Column '{name}' (Type: {dtype}, Range: {min_val} to {max_val})")
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM data_table;")
    row_count = cursor.fetchone()[0]
    
    # Get sample values for each column
    sample_values = {}
    for col in columns:
        name = col[1]
        cursor.execute(f"SELECT DISTINCT {name} FROM data_table LIMIT 5;")
        samples = cursor.fetchall()
        sample_values[name] = [str(s[0]) for s in samples]
    
    conn.close()
    
    table_info = f"""
    Table Name: data_table
    Total Rows: {row_count}
    
    Columns and Data Types:
    {'\n'.join(schema_info)}
    
    Sample Values:
    {'\n'.join(f"{col}: {', '.join(vals)}" for col, vals in sample_values.items())}
    """
    
    return table_info

def analyze_query_intent(question: str, table_info: str) -> bool:
    """Analyze if the question can be answered using the available data"""
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    intent_prompt = f"""
    Given the following database information and user question, determine if the question can be answered using the available data.
    Return only 'yes' or 'no'.
    
    Database Information:
    {table_info}
    
    User Question: {question}
    
    Can this question be answered using the available data (yes/no)?:
    """
    
    response = chat_model.invoke(intent_prompt).content.strip().lower()
    return response == 'yes'

def clean_sql_query(query: str) -> str:
    """Clean and extract SQL query from the model's response"""
    query = re.sub(r'```sql|```', '', query)
    query = query.strip().rstrip(';')
    return query

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
    You are a SQL expert. Given the following database information and question, generate a SQL query if possible.
    If the question cannot be answered using SQL or the available data, respond with "I cannot answer this question with the available data."
    If you generate a SQL query, return ONLY the SQL query without any explanations or decorations.
    
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
            # First, analyze if the question can be answered with available data
            can_answer = analyze_query_intent(question, st.session_state.table_info)
            
            if not can_answer:
                response = "I cannot answer this question with the available data."
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Get relevant context from vector store
                docs = st.session_state.vector_store.similarity_search(question)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Generate SQL query
                sql_query = chain.run(context=context, question=question)
                
                # Check if the response indicates inability to answer
                if "cannot answer" in sql_query.lower():
                    st.session_state.messages.append({"role": "assistant", "content": sql_query})
                else:
                    # Clean and execute the query
                    cleaned_query = clean_sql_query(sql_query)
                    results = execute_sql_query(st.session_state.db_path, cleaned_query)
                    
                    # Format response
                    response = f"SQL Query:\n```sql\n{cleaned_query}\n```\n\nResults:\n{results.to_markdown()}"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            error_message = f"Error: {str(e)}"
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
