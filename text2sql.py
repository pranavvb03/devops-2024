import streamlit as st
st.set_page_config(layout="wide")
import atexit
import numpy as np
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
from typing import List, Dict, Any
from datetime import datetime
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# Initialize session state variables
session_state_keys = [
    'messages', 'db_path', 'vector_store', 'table_info', 'chat_history', 
    'current_chat', 'df_preview', 'query_history', 'favorites',
    'query_explanation', 'execution_time', 'query_templates'
]

for key in session_state_keys:
    if key not in st.session_state:
        if key in ['messages', 'chat_history', 'query_history', 'favorites', 'query_templates']:
            st.session_state[key] = []
        else:
            st.session_state[key] = None

# Initialize finance-specific query templates
if not st.session_state.query_templates:
    st.session_state.query_templates = [
        {"name": "Transaction Analysis", "template": "Show me the top {n} transactions by {amount_column}"},
        {"name": "Fraud Detection", "template": "Find suspicious transactions where {amount_column} > {threshold} and {time_column} is during non-business hours"},
        {"name": "Revenue Trend", "template": "Show {revenue_column} trend over {time_column}"},
        {"name": "Expense Breakdown", "template": "Summarize {expense_column} by {category_column}"},
        {"name": "ROI Analysis", "template": "Calculate ROI where ROI = ({revenue_column} - {cost_column}) / {cost_column} grouped by {investment_type}"},
        {"name": "Customer Profitability", "template": "Calculate profit margin (({revenue_column} - {cost_column}) / {revenue_column} * 100) by {customer_column}"},
        {"name": "Budget Variance", "template": "Compare {actual_column} vs {budget_column} and calculate variance percentage"}
    ]

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

def save_to_favorites(query, result=None):
    """Save query to favorites"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    favorite = {
        'timestamp': timestamp,
        'query': query,
        'result': result.to_dict('records') if result is not None else None
    }
    st.session_state.favorites.append(favorite)
    return "Query saved to favorites!"

def create_db_from_csv(csv_file) -> str:
    """Create SQLite database from uploaded CSV file"""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_db.name
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Save a preview of the dataframe
    st.session_state.df_preview = df.head(10)
    
    # Clean column names: remove spaces and special characters
    df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
    
    # Create SQLite connection
    conn = sqlite3.connect(db_path)
    
    # Save DataFrame to SQLite
    table_name = 'financial_data'
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    conn.close()
    return db_path

def get_table_info(db_path: str) -> str:
    """Get comprehensive table information"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("PRAGMA table_info(financial_data);")
    columns = cursor.fetchall()
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM financial_data;")
    row_count = cursor.fetchone()[0]
    
    # Get column information and sample data
    column_info = []
    column_stats = []
    
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        
        # Get distinct count
        cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM financial_data;")
        distinct_count = cursor.fetchone()[0]
        
        # Get samples
        cursor.execute(f"SELECT {col_name} FROM financial_data LIMIT 3;")
        samples = [str(row[0]) for row in cursor.fetchall()]
        
        # Get min, max for numeric columns
        min_val = max_val = avg_val = None
        if col_type in ['INTEGER', 'REAL', 'FLOAT', 'DOUBLE']:
            try:
                cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM financial_data;")
                min_val, max_val, avg_val = cursor.fetchone()
                stats = f"Min: {min_val}, Max: {max_val}, Avg: {round(avg_val, 2) if avg_val is not None else 'N/A'}"
            except:
                stats = "Stats not available"
        else:
            stats = "Non-numeric column"
        
        column_info.append(f"Column '{col_name}' (Type: {col_type}, Distinct Values: {distinct_count}, Examples: {', '.join(samples)})")
        column_stats.append({
            "name": col_name,
            "type": col_type,
            "distinct_count": distinct_count,
            "samples": samples,
            "stats": stats
        })
    
    conn.close()
    
    return f"""
    Table Name: financial_data
    Total Rows: {row_count}
    
    Schema Information:
    {'\n'.join(column_info)}
    """, column_stats

def get_query_explanation(query: str) -> str:
    """Get natural language explanation of SQL query with financial context"""
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    explain_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Explain the following SQL query in simple financial terms that a financial analyst or banker would understand:
        
        ```sql
        {query}
        ```
        
        Break down each component, explain what financial insights it provides, and how it could be used for financial analysis.
        """
    )
    
    explain_chain = LLMChain(llm=chat_model, prompt=explain_prompt)
    
    try:
        explanation = explain_chain.run(query=query)
        return explanation
    except Exception as e:
        return f"Could not generate explanation: {str(e)}"

def clean_sql_query(query: str) -> str:
    """Clean and extract SQL query from the model's response"""
    # Remove markdown code block formatting (backticks)
    query = re.sub(r'^```sql|^```|```$', '', query, flags=re.MULTILINE)
    
    # Remove any trailing semicolons and extra whitespace
    query = query.strip().rstrip(';')
    
    # Add semicolon back for consistency
    return query + ';'

def execute_sql_query(db_path: str, query: str) -> tuple:
    """Execute SQL query and return results as DataFrame along with execution time"""
    conn = sqlite3.connect(db_path)
    start_time = time.time()
    
    try:
        df = pd.read_sql_query(query, conn)
        execution_time = round(time.time() - start_time, 4)
        
        # Save to query history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.query_history.append({
            'timestamp': timestamp,
            'query': query,
            'execution_time': execution_time,
            'row_count': len(df)
        })
        
        conn.close()
        return df, execution_time
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

def create_financial_chart(df, chart_type):
    """Create financial visualization based on dataframe and chart type"""
    try:
        if df.empty or len(df.columns) < 1:
            return None, "Cannot create chart: Not enough data"
        
        # Try to identify good candidates for x and y axes
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # For charts that need category/time data for x-axis
        x_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
        
        # For charts that need numeric data for y-axis
        y_col = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Financial chart types
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"Financial Analysis: {y_col} by {x_col}")
            return fig, None
        
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=f"Trend Analysis: {y_col} over {x_col}")
            return fig, None
        
        elif chart_type == "pie":
            if len(numeric_cols) > 0:
                fig = px.pie(df, names=x_col, values=y_col, title=f"Distribution: {y_col} by {x_col}")
                return fig, None
            else:
                return None, "Cannot create pie chart: No numeric columns found for values"
        
        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                              title=f"Correlation Analysis: {numeric_cols[1]} vs {numeric_cols[0]}")
                return fig, None
            else:
                return None, "Cannot create scatter plot: Need at least 2 numeric columns"
        
        elif chart_type == "heatmap":
            if len(numeric_cols) >= 1 and len(non_numeric_cols) >= 2:
                pivot = df.pivot_table(
                    values=numeric_cols[0], 
                    index=non_numeric_cols[0], 
                    columns=non_numeric_cols[1], 
                    aggfunc='mean'
                )
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='Viridis'
                ))
                fig.update_layout(title=f"Risk Analysis: {numeric_cols[0]} by {non_numeric_cols[0]} and {non_numeric_cols[1]}")
                return fig, None
            else:
                return None, "Cannot create heatmap: Need at least 1 numeric column and 2 categorical columns"
        
        else:
            return None, f"Unsupported chart type: {chart_type}"
    
    except Exception as e:
        return None, f"Error creating chart: {str(e)}"

def detect_visualization_request(question):
    """Detect if user is asking for visualization"""
    vis_keywords = [
        "visualize", "visualization", "plot", "chart", "graph", "diagram",
        "show me a chart", "display a graph", "create a plot", "draw a",
        "trend", "compare", "distribution", "breakdown"
    ]
    
    question_lower = question.lower()
    for keyword in vis_keywords:
        if keyword in question_lower:
            return True
    return False

def identify_chart_type(question):
    """Identify the type of chart requested"""
    question_lower = question.lower()
    
    chart_types = {
        "bar": ["bar chart", "bar graph", "column chart", "breakdown", "compare"],
        "pie": ["pie chart", "pie graph", "distribution", "allocation", "percentage"],
        "line": ["line chart", "line graph", "trend", "over time", "historical"],
        "scatter": ["scatter plot", "scatter graph", "correlation", "relationship"],
        "heatmap": ["heatmap", "heat map", "risk matrix", "exposure"]
    }
    
    for chart_type, keywords in chart_types.items():
        for keyword in keywords:
            if keyword in question_lower:
                return chart_type
    
    # Default to bar chart if visualization is requested but type is not specified
    return "bar"

def extract_sql_refinement_intent(question):
    """Determine if user wants to refine previous SQL query"""
    refinement_keywords = [
        "modify", "change", "refine", "update", "adjust", "previous query",
        "last query", "that query", "fix", "improve", "edit", "adapt"
    ]
    
    question_lower = question.lower()
    for keyword in refinement_keywords:
        if keyword in question_lower:
            return True
    return False

def generate_follow_up_questions(context, question, result_df):
    """Generate finance-specific follow-up questions based on current question and results"""
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    
    # Create a description of the result data
    result_description = "No results" if result_df is None or result_df.empty else f"Result with {len(result_df)} rows and columns: {', '.join(result_df.columns.tolist())}"
    
    follow_up_prompt = PromptTemplate(
        input_variables=["context", "question", "result_description"],
        template="""
        As a financial analyst, based on the following database information, the user's current question, and the query results,
        suggest 3 finance-specific follow-up questions a banker or financial analyst might want to ask next. Make them conversational and directly
        related to the current question or results. Focus on financial insights like risk analysis, profitability, trends, or anomalies.
        Return them as a comma-separated list.
        
        Database Information:
        {context}
        
        User's Current Question: {question}
        
        Query Result Information: {result_description}
        
        3 Finance-Specific Follow-up Questions (comma-separated):
        """
    )
    
    follow_up_chain = LLMChain(llm=chat_model, prompt=follow_up_prompt)
    
    try:
        follow_ups = follow_up_chain.run(
            context=context, 
            question=question, 
            result_description=result_description
        )
        return [q.strip() for q in follow_ups.split(',') if q.strip()]
    except Exception as e:
        return ["Would you like to see a trend analysis of this data?", 
                "Should we look for potential fraud indicators in these transactions?", 
                "Would you like to compare this to the previous quarter's performance?"]

# Create Streamlit interface
st.title("Finance & Banking SQL Assistant")
st.write("Upload financial transaction data to analyze and gain insights without SQL expertise!")

# Sidebar for navigation and features
with st.sidebar:
    st.title("Navigation")
    menu_choice = st.radio("Menu", ["New Analysis", "Analysis History", "Query Library", "Favorites", "Templates", "About"])
    
    if menu_choice == "Analysis History":
        st.subheader("Previous Analyses")
        for idx, chat in enumerate(st.session_state.chat_history):
            if st.button(f"Analysis {idx + 1} - {chat['timestamp']}"):
                load_chat(chat)
        
        if st.button("Clear History"):
            st.session_state.chat_history = []
    
    elif menu_choice == "Query Library":
        st.subheader("Recent Queries")
        for idx, query in enumerate(st.session_state.query_history):
            with st.expander(f"Query {idx + 1} - {query['timestamp']}"):
                st.code(query['query'], language="sql")
                st.text(f"Execution time: {query['execution_time']}s, Rows: {query['row_count']}")
                if st.button(f"Reuse Query #{idx + 1}"):
                    st.session_state.messages.append({"role": "user", "content": f"Execute this SQL: {query['query']}"})
        
        if st.button("Clear Query Library"):
            st.session_state.query_history = []
    
    elif menu_choice == "Favorites":
        st.subheader("Favorite Queries")
        for idx, fav in enumerate(st.session_state.favorites):
            with st.expander(f"Favorite {idx + 1} - {fav['timestamp']}"):
                st.code(fav['query'], language="sql")
                if st.button(f"Reuse Favorite #{idx + 1}"):
                    st.session_state.messages.append({"role": "user", "content": f"Execute this SQL: {fav['query']}"})
        
        if st.button("Clear Favorites"):
            st.session_state.favorites = []
    
    elif menu_choice == "Templates":
        st.subheader("Financial Analysis Templates")
        for idx, template in enumerate(st.session_state.query_templates):
            with st.expander(f"{template['name']}"):
                st.text(template['template'])
                if st.button(f"Use Template #{idx + 1}"):
                    st.session_state.messages.append({"role": "user", "content": template['template']})
    
    elif menu_choice == "About":
        st.markdown("""
        ### Finance & Banking SQL Assistant
        
        This tool helps financial analysts and bankers:
        
        - Analyze transaction records without SQL expertise
        - Detect potential fraud patterns
        - Generate insights from investment data
        - Track financial performance metrics
        - Visualize financial trends
        - Create financial reports
        - Identify outliers and anomalies
        - Calculate key financial ratios
        
        Simply upload your financial data and ask questions in plain English!
        """)

# Main content area
if menu_choice == "New Analysis" or menu_choice == "Analysis History":
    # File upload
    uploaded_file = st.file_uploader("Upload financial data (CSV)", type="csv")
    
    if uploaded_file is not None and st.session_state.db_path is None:
        st.session_state.db_path = create_db_from_csv(uploaded_file)
        st.session_state.table_info, column_stats = get_table_info(st.session_state.db_path)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vector_store = create_vector_store(st.session_state.table_info, embeddings)
        st.success("Financial database created successfully!")
        
    # Database info expander section
    if st.session_state.db_path:
        with st.expander("Financial Data Overview"):
            tabs = st.tabs(["Data Preview", "Schema", "Statistics"])
            
            with tabs[0]:  # Preview tab
                if st.session_state.df_preview is not None:
                    st.dataframe(st.session_state.df_preview)
            
            with tabs[1]:  # Schema tab
                if st.session_state.table_info:
                    st.text(st.session_state.table_info)
            
            with tabs[2]:  # Statistics tab
                if st.session_state.db_path:
                    conn = sqlite3.connect(st.session_state.db_path)
                    df = pd.read_sql_query("SELECT * FROM financial_data", conn)
                    conn.close()
                    
                    st.write("Financial Metrics Summary:")
                    st.dataframe(df.describe())

    if st.session_state.db_path:           
        chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["context", "question", "history"],
            template="""
            You are a financial SQL expert. Generate a SQL query based on the following financial database information, conversation history, and question.
            Focus on providing financial insights, detecting patterns, analyzing trends, and identifying anomalies.
            Return ONLY the SQL query without any explanations or decorations.
            If you cannot generate a valid query, respond with "I cannot answer this question with the available financial data."
            
            Financial Database Information:
            {context}
            
            Previous Conversation:
            {history}
            
            User Question: {question}
            
            Response:
            """
        )
        
        chain = LLMChain(llm=chat_model, prompt=prompt)
        
        # Chat interface with history management
        col1, col2 = st.columns([3, 1])
        with col1:
            if question := st.chat_input("Ask a financial question in plain English..."):
                # Check if this is a refinement request
                is_refinement = extract_sql_refinement_intent(question)
                
                # Get previous SQL query if refinement is requested
                previous_query = None
                if is_refinement:
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "assistant" and "SQL Query:" in msg["content"]:
                            query_match = re.search(r'```sql\n(.*?)\n```', msg["content"], re.DOTALL)
                            if query_match:
                                previous_query = query_match.group(1)
                                break
                
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Create chat history context for the model
                history_text = ""
                for msg in st.session_state.messages[-6:-1]:  # Last 5 messages excluding current
                    if msg["role"] == "user":
                        history_text += f"User: {msg['content']}\n"
                    else:
                        # Extract just the query part if it's an assistant message with SQL
                        if "SQL Query:" in msg["content"]:
                            query_match = re.search(r'```sql\n(.*?)\n```', msg["content"], re.DOTALL)
                            if query_match:
                                history_text += f"Assistant: Generated SQL: {query_match.group(1)}\n"
                            else:
                                history_text += f"Assistant: {msg['content']}\n"
                        else:
                            history_text += f"Assistant: {msg['content']}\n"
                
                try:
                    # Get relevant database context
                    docs = st.session_state.vector_store.similarity_search(question)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    if is_refinement and previous_query:
                        question = f"Refine this financial SQL query: {previous_query}\nNew requirements: {question}"
                    
                    # Generate SQL query
                    sql_response = chain.run(context=context, question=question, history=history_text)
                    
                    if "cannot answer" in sql_response.lower():
                        st.session_state.messages.append({"role": "assistant", "content": sql_response})
                    else:
                        cleaned_query = clean_sql_query(sql_response)
                        
                        # Get query explanation with financial context
                        explanation = get_query_explanation(cleaned_query)
                        
                        # Execute query with timing
                        results, exec_time = execute_sql_query(st.session_state.db_path, cleaned_query)
                        
                        # Generate follow-up questions
                        follow_up_questions = generate_follow_up_questions(context, question, results)
                        
                        # Check if visualization is requested
                        viz_requested = detect_visualization_request(question)
                        
                        if viz_requested:
                            chart_type = identify_chart_type(question)
                            fig, error = create_financial_chart(results, chart_type)
                            
                            if fig:
                                # Build response with query details
                                response = f"""
SQL Query:
```sql
{cleaned_query}
```

Execution Time: {exec_time}s

Results:
{results.to_markdown()}

Financial Insights:
{explanation}
                                """
                                
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                
                                # Store chart information in the message
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": f"Here's a financial {chart_type} visualization of your data:",
                                    "chart": {
                                        "type": chart_type,
                                        "data": results.to_dict('records')
                                    }
                                })
                                
                                # Add follow-up questions if available
                                if follow_up_questions:
                                    follow_up_text = "You might want to ask:\n\n" + "\n".join([f"- {q}" for q in follow_up_questions])
                                    st.session_state.messages.append({"role": "assistant", "content": follow_up_text})
                                
                            else:
                                response = f"""
SQL Query:
```sql
{cleaned_query}
```

Execution Time: {exec_time}s

Results:
{results.to_markdown()}

Financial Insights:
{explanation}

Couldn't create visualization: {error}
                                """
                                st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            # Build response with query details
                            response = f"""
SQL Query:
```sql
{cleaned_query}
```

Execution Time: {exec_time}s

Results:
{results.to_markdown()}

Financial Insights:
{explanation}
                            """
                            
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Add follow-up questions if available
                            if follow_up_questions:
                                follow_up_text = "You might want to ask:\n\n" + "\n".join([f"- {q}" for q in follow_up_questions])
                                st.session_state.messages.append({"role": "assistant", "content": follow_up_text})
                        
                        save_chat_history()
                        
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
        
        with col2:
            col2_btns = st.columns(2)
            with col2_btns[0]:
                if st.button("Clear Analysis"):
                    clear_chat()
            with col2_btns[1]:
                if st.button("Save Query"):
                    # Find the last SQL query in the chat
                    last_query = None
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "assistant" and "SQL Query:" in msg["content"]:
                            query_match = re.search(r'```sql\n(.*?)\n```', msg["content"], re.DOTALL)
                            if query_match:
                                last_query = query_match.group(1)
                                save_msg = save_to_favorites(last_query)
                                st.success(save_msg)
                                break
                    if not last_query:
                        st.error("No SQL query found to save")
        
        # Display chat with visualization support
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If the message has a chart, display it
                if "chart" in message:
                    chart_data = pd.DataFrame.from_records(message["chart"]["data"])
                    chart_type = message["chart"]["type"]
                    fig, _ = create_financial_chart(chart_data, chart_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

# Cleanup
def cleanup():
    if st.session_state.db_path and os.path.exists(st.session_state.db_path):
        os.unlink(st.session_state.db_path)

atexit.register(cleanup)
