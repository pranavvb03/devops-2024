import streamlit as st
st.set_page_config(layout="wide")
import atexit
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
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt

# Initialize session state
for key in ['messages', 'db_path', 'vector_store', 'table_info', 'chat_history', 'current_chat']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['messages', 'chat_history'] else None

# Set your Google API key
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

def save_chat_history():
    """Save current chat to history"""
    if st.session_state.messages:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat = {
            'timestamp': timestamp,
            'messages': st.session_state.messages,
            'table_info': st.session_state.table_info
        }
        st.session_state.chat_history.append(chat)

def load_chat(chat):
    """Load selected chat"""
    st.session_state.messages = chat['messages']
    st.session_state.table_info = chat['table_info']

def clear_chat():
    """Clear current chat"""
    st.session_state.messages = []

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
    query = re.sub(r'```sql|```', '', query)
    query = query.strip().rstrip(';')
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

st.title("Interactive SQL Chatbot with RAG")
st.write("Upload a CSV file to create a database and ask questions in natural language!")

with st.sidebar:
    st.title("Navigation")
    menu_choice = st.radio("Menu", ["New Chat", "Chat History", "About"])
    if menu_choice == "Chat History":
        st.subheader("Previous Chats")
        for idx, chat in enumerate(st.session_state.chat_history):
            if st.button(f"Chat {idx + 1} - {chat['timestamp']}"):
                load_chat(chat)
        if st.button("Clear All History"):
            st.session_state.chat_history = []
    elif menu_choice == "About":
        st.markdown("""
        ### Text-to-SQL Chatbot
        - Upload CSV files
        - Ask questions in natural language
        - Get SQL queries and results
        - Maintains chat history
        """)

if menu_choice == "New Chat" or menu_choice == "Chat History":
    st.title("SQL Chatbot")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None and st.session_state.db_path is None:
        st.session_state.db_path = create_db_from_csv(uploaded_file)
        st.session_state.table_info = get_table_info(st.session_state.db_path)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vector_store = create_vector_store(st.session_state.table_info, embeddings)
        st.success("Database created!")

    if st.session_state.db_path:
        chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", temperature=0.3)

        if question := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": question})
            try:
                docs = st.session_state.vector_store.similarity_search(question)
                context = "\n".join([doc.page_content for doc in docs])
                sql_response = chain.run(context=context, question=question)
                cleaned_query = clean_sql_query(sql_response)
                results = execute_sql_query(st.session_state.db_path, cleaned_query)
                st.dataframe(results)
                if st.button("Show Chart"):
                    fig, ax = plt.subplots()
                    results.plot(kind='bar', ax=ax)
                    st.pyplot(fig)
                    pie_fig, pie_ax = plt.subplots()
                    results.iloc[:, 0].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=pie_ax, colors=plt.cm.Paired.colors)
                    st.pyplot(pie_fig)
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

atexit.register(lambda: os.unlink(st.session_state.db_path) if st.session_state.db_path and os.path.exists(st.session_state.db_path) else None)

