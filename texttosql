import streamlit as st
import sqlite3
import pandas as pd
from io import StringIO
import google.generativeai as genai
from google.generativeai import chat

# Function to initialize Google Gemini Pro API (replace with your credentials)
def initialize_google_gemini():
    GEMINI_API_KEY = st.secrets["YOUR_GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)

# Function to parse and execute SQL query against SQLite DB
def execute_sql_query(db_connection, query):
    try:
        cursor = db_connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(results, columns=columns)
    except Exception as e:
        return f"Error: {e}"

# Streamlit App
st.title("Text-to-SQL Chatbot")
st.write("Upload a CSV file to create a database, and ask questions about the data.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

db_connection = None
if uploaded_file:
    try:
        # Load CSV to DataFrame
        csv_data = pd.read_csv(uploaded_file)

        # Create SQLite Database
        db_connection = sqlite3.connect("uploaded_database.db")
        table_name = "data"
        csv_data.to_sql(table_name, db_connection, if_exists="replace", index=False)

        st.success("Database created successfully!")
        st.write("Schema of the uploaded data:")
        st.dataframe(csv_data.head())

    except Exception as e:
        st.error(f"Failed to process the uploaded CSV file. Error: {e}")

# Chatbot Interaction
if db_connection:
    initialize_google_gemini()

    st.write("Ask questions about the uploaded data:")
    user_query = st.text_input("Enter your query in plain English:")

    if st.button("Generate SQL and Fetch Results"):
        if user_query.strip():
            try:
                # Generate SQL query using Google Gemini Pro
                gemini_response = chat.complete(
                    prompt=f"The database has a table named '{table_name}' with schema: {list(csv_data.columns)}. Translate the following question into a MySQL query: {user_query}",
                    model="gemini-1.5-pro"
                )
                sql_query = gemini_response["text"]

                st.write("Generated SQL Query:")
                st.code(sql_query)

                # Execute SQL query
                results = execute_sql_query(db_connection, sql_query)

                if isinstance(results, pd.DataFrame):
                    st.write("Query Results:")
                    st.dataframe(results)
                else:
                    st.error(results)

            except Exception as e:
                st.error(f"Error in generating or executing query: {e}")
        else:
            st.warning("Please enter a query in plain English.")

# Close database connection when app ends
if db_connection:
    db_connection.close()
