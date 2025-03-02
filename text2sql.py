import streamlit as st
st.set_page_config(layout="wide")
import atexit
import pandas as pd
import re
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
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
    if st.session_state.messages:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat = {
            'timestamp': timestamp,
            'messages': st.session_state.messages,
            'table_info': st.session_state.table_info
        }
        st.session_state.chat_history.append(chat)

def load_chat(chat):
    st.session_state.messages = chat['messages']
    st.session_state.table_info = chat['table_info']

def clear_chat():
    st.session_state.messages = []

def create_db_from_csv(csv_file) -> str:
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_db.name
    df = pd.read_csv(csv_file)
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    conn = sqlite3.connect(db_path)
    table_name = 'data_table'
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    conn.close()
    return db_path

def get_table_info(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(data_table);")
    columns = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) FROM data_table;")
    row_count = cursor.fetchone()[0]
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
    return f"Table Name: data_table\nTotal Rows: {row_count}\n\nSchema Information:\n{''.join(column_info)}"

def execute_sql_query(db_path: str, query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        raise e

def display_chart(dataframe: pd.DataFrame):
    if not dataframe.empty:
        st.write("Select columns for visualization")
        x_col = st.selectbox("X-axis", dataframe.columns)
        y_col = st.selectbox("Y-axis", dataframe.columns)
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

        plt.figure(figsize=(10, 6))
        if chart_type == "Bar":
            plt.bar(dataframe[x_col], dataframe[y_col])
        elif chart_type == "Line":
            plt.plot(dataframe[x_col], dataframe[y_col])
        elif chart_type == "Scatter":
            plt.scatter(dataframe[x_col], dataframe[y_col])

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{chart_type} Chart of {y_col} vs {x_col}")
        st.pyplot(plt)

st.title("Interactive SQL Chatbot with RAG")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and st.session_state.db_path is None:
    st.session_state.db_path = create_db_from_csv(uploaded_file)
    st.session_state.table_info = get_table_info(st.session_state.db_path)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vector_store = create_vector_store(st.session_state.table_info, embeddings)
    st.success("Database created!")

if st.session_state.db_path:
    question = st.chat_input("Ask a question")
    if question:
        chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a SQL expert. Generate a SQL query based on the following database information and question.
            Return ONLY the SQL query without any explanations or decorations.
            If you cannot generate a valid query, respond with "I cannot answer this question with the available data."
            Database Information:
            {context}
            User Question: {question}
            Response:
            """
        )
        chain = LLMChain(llm=chat_model, prompt=prompt)
        docs = st.session_state.vector_store.similarity_search(question)
        context = "\n".join([doc.page_content for doc in docs])
        sql_response = chain.run(context=context, question=question)

        if "cannot answer" not in sql_response.lower():
            cleaned_query = sql_response.strip().rstrip(';') + ';'
            results = execute_sql_query(st.session_state.db_path, cleaned_query)
            st.write(results)
            if st.checkbox("Visualize Results"):
                display_chart(results)

atexit.register(lambda: os.unlink(st.session_state.db_path) if st.session_state.db_path and os.path.exists(st.session_state.db_path) else None)

