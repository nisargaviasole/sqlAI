import os
import json
import pyodbc
import psycopg2
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import dotenv_values
import sys
from streamlit.watcher import local_sources_watcher
import uuid
from datetime import datetime, date
from decimal import Decimal
import uuid
import re

# Patch LocalSourcesWatcher to skip torch.classes
original_get_module_paths = local_sources_watcher.get_module_paths


def patched_get_module_paths(module):
    if module.__name__.startswith("torch.classes"):
        return []
    return original_get_module_paths(module)


local_sources_watcher.get_module_paths = patched_get_module_paths

# Load environment variables
env_vars = dotenv_values(".env")
GROQ_API_KEY = env_vars.get("GROQ_API_KEY")
connection_string = env_vars.get("SQL_DATABASE")

# Initialize embedding model
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    cache_folder=os.path.join(os.getcwd(), "hf_cache"),
    model_kwargs={"device": "cpu"},
)

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
)


# Load credentials from JSON file
def load_credentials(filename="credentials.json"):
    try:
        with open(filename, "r") as f:
            credentials = json.load(f)
        return credentials
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading credentials from {filename}: {e}")
        return []

def load_schemas(filename="schemas.json"):
    try:
        with open(filename, "r") as f:
            schemas = json.load(f)
        return schemas
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading schemas from {filename}: {e}")
        return []
# PostgreSQL connection
# def get_db_connection(pg_host, pg_database, pg_user, pg_password, pg_port):
    try:
        conn = psycopg2.connect(
            host=pg_host,
            database=pg_database,
            user=pg_user,
            password=pg_password,
            port=pg_port,
        )
        print("database connected")
        return conn
    except psycopg2.Error as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None


def get_sqlserver_connection(conn_str):
    try:
        conn = pyodbc.connect(conn_str)
        print("SQL Server connected")
        return conn
    except pyodbc.Error as e:
        st.error(f"Error connecting to SQL Server: {e}")
        return None


# def extract_postgres_schema(conn):
    """
    Extract schema (tables, columns, relationships) from PostgreSQL database.
    Returns a list of dictionaries compatible with NLtoSQL.initialize_schema.
    """
    if not conn:
        raise Exception("No database connection")

    schema_data = []
    try:
        with conn.cursor() as cur:
            # Get all tables in the public schema
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """
            )
            tables = [row[0] for row in cur.fetchall()]

            for table_name in tables:
                # Get table description (comment)
                cur.execute(
                    """
                    SELECT obj_description(t.oid, 'pg_class')
                    FROM pg_class t
                    WHERE t.relname = %s AND t.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
                """,
                    (table_name,),
                )
                table_description = cur.fetchone()[0] or ""

                # Get columns and their details
                cur.execute(
                    """
                    SELECT 
                        column_name,
                        data_type,
                        col_description((SELECT oid FROM pg_class WHERE relname = %s), ordinal_position)
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                """,
                    (table_name, table_name),
                )
                columns = []
                column_details = {}
                for col_name, data_type, col_description in cur.fetchall():
                    columns.append(col_name)
                    column_details[col_name] = {
                        "data_type": data_type.upper(),
                        "description": col_description or "",
                    }

                # Get relationships (foreign keys)
                cur.execute(
                    """
                    SELECT 
                        tc.constraint_name,
                        kcu.column_name AS from_column,
                        ccu.table_name AS related_table,
                        ccu.column_name AS to_column
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_schema = 'public'
                        AND tc.table_name = %s
                """,
                    (table_name,),
                )
                relationships = []
                for _, from_column, related_table, to_column in cur.fetchall():
                    relationships.append(
                        {
                            "type": "many-to-one",
                            "related_table": related_table,
                            "from_column": from_column,
                            "to_column": to_column,
                        }
                    )

                # Construct schema entry
                table_info = {
                    "table_name": table_name,
                    "description": table_description,
                    "columns": columns,
                    "column_details": column_details,
                    "relationships": relationships,
                }
                schema_data.append(table_info)

    finally:
        pass  # Connection will be managed by session state

    return schema_data


def extract_sqlserver_schema(conn):
    schema_data = []

    cursor = conn.cursor()

    # Get tables
    cursor.execute("""
        SELECT t.name
        FROM sys.tables t
        WHERE t.is_ms_shipped = 0
    """)
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        # Get columns
        cursor.execute("""
            SELECT c.name, ty.name
            FROM sys.columns c
            JOIN sys.types ty ON c.user_type_id = ty.user_type_id
            WHERE c.object_id = OBJECT_ID(?)
        """, table)

        columns = []
        column_details = {}

        for col, dtype in cursor.fetchall():
            columns.append(col)
            column_details[col] = {
                "data_type": dtype.upper(),
                "description": ""
            }

        # Get foreign keys
        cursor.execute("""
            SELECT 
                OBJECT_NAME(fkc.parent_object_id) AS table_name,
                COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS from_column,
                OBJECT_NAME(fkc.referenced_object_id) AS related_table,
                COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS to_column
            FROM sys.foreign_key_columns fkc
            WHERE OBJECT_NAME(fkc.parent_object_id) = ?
        """, table)

        relationships = []
        for _, from_col, rel_table, to_col in cursor.fetchall():
            relationships.append({
                "type": "many-to-one",
                "related_table": rel_table,
                "from_column": from_col,
                "to_column": to_col,
            })

        schema_data.append({
            "table_name": table,
            "description": "",
            "columns": columns,
            "column_details": column_details,
            "relationships": relationships,
        })

    return schema_data


def save_schema_to_json(schema_data, filename="schemas.json"):
    """
    Save the schema data to a JSON file.
    If the file already exists, it is deleted first.
    """
    try:
        # Delete the file if it exists
        if os.path.exists(filename):
            os.remove(filename)

        # Write the new schema data
        with open(filename, "w") as f:
            json.dump(schema_data, f, indent=4)

        st.success(f"Schema saved to {filename}")
    except Exception as e:
        st.error(f"Error saving schema to {filename}: {e}")


def load_schema_from_json(filename="schemas.json"):
    """
    Load schema data from a JSON file.
    Returns None if the file doesn't exist or is invalid.
    """
    try:
        with open(filename, "r") as f:
            schema_data = json.load(f)
        st.success(f"Schema loaded from {filename}")
        return schema_data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Error loading schema from {filename}: {e}")
        return None


def execute_sql(conn, sql_query, max_rows=20):
    cursor = conn.cursor()
    cursor.execute(sql_query)

    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchmany(max_rows)

    results = [dict(zip(columns, row)) for row in rows]
    return results

def make_json_safe(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return str(obj)

def sanitize_sql(sql: str) -> str:
    sql = sql.strip()

    # Remove markdown blocks
    sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)

    # Remove backticks (MySQL style)
    sql = sql.replace("`", "")

    # Keep only first SQL statement
    if ";" in sql:
        sql = sql.split(";")[0] + ";"

    return sql.strip()


class NLtoSQL:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None

    def initialize_schema(self, schema_data):
        """
        Load database schema into FAISS vector store.
        schema_data should be a list of dictionaries with table and column information.
        """
        documents = []
        for table_info in schema_data:
            table_name = table_info["table_name"]
            columns = table_info["columns"]

            table_doc = f"Table: {table_name}\nDescription: {table_info.get('description', '')}\nColumns: {', '.join(columns)}"
            documents.append(table_doc)

            for col in columns:
                col_info = table_info.get("column_details", {}).get(col, {})
                col_doc = f"Column: {col} in Table: {table_name}\nData Type: {col_info.get('data_type', 'unknown')}\nDescription: {col_info.get('description', '')}"
                documents.append(col_doc)

            for rel in table_info.get("relationships", []):
                rel_doc = f"Relationship: Table {table_name} has a {rel['type']} relationship with {rel['related_table']} via columns {rel['from_column']} and {rel['to_column']}"
                documents.append(rel_doc)

        # Create FAISS vector store from schema texts
        self.vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        st.success(f"Loaded {len(documents)} schema documents into FAISS")

    def process_natural_language(self, query):
        retrieved_docs = self.retriever.invoke(query)
        print(retrieved_docs)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"Context : {context}")
        system_prompt = f"""
            You are a SQL Server (T-SQL) query generator.

            Convert natural language into VALID SQL Server queries using the schema below.

            Schema:
            {context}

            Rules:
            - Return ONLY the SQL query
            - Use SQL Server syntax (T-SQL)
            - Use [TableName] and [ColumnName] with square brackets
            - Use LIKE instead of ILIKE
            - Use INNER JOIN by default
            - Use LEFT JOIN only if explicitly required
            - Do NOT use PostgreSQL features
            - Avoid subqueries when JOINs are possible
            - If user mentions a name (e.g., "dev"), assume it's stored in a related table and JOIN properly
            - If schema is insufficient, return:
            - Additional information needed
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate a SQL query for: {query}"),
        ]

        response = llm.invoke(messages)
        sql_query = response.content.strip()

        if sql_query.startswith("```sql"):
            sql_query = sql_query[7:].strip()
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3].strip()

        return sql_query

    def rephrase_result(self, user_query, sql_query, results):
        system_prompt = """
            You are a data analyst assistant.

            Given:
            - A user's question
            - The SQL query used
            - The SQL result data

            Convert the result into a clear, concise, human-friendly answer.
            If result is empty, say no data was found.
            Avoid technical jargon.
            Dont give detailed information just give a response as you chat with the user
            Give answer in 50 words maximum and if its required to give details then max 100
            """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
                User question:
                {user_query}

                SQL query:
                {sql_query}

                SQL result:
                {json.dumps(results, indent=2, default=make_json_safe)}
            """)
        ]

        response = llm.invoke(messages)
        return response.content.strip()


def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.title("Natural Language to SQL Query Generator")
    st.write(
        "Click on Connect Button for connect with database"
    )

    # Initialize session state variables
    if "db_connection" not in st.session_state:
        st.session_state.db_connection = None
    if "nl_to_sql" not in st.session_state:
        st.session_state.nl_to_sql = None
    if "connected" not in st.session_state:
        st.session_state.connected = False

    # Database connection form
    st.subheader("Database Connection")
    if st.button("Connect" if not st.session_state.connected else "Disconnect"):

        # -------- Disconnect --------
        if st.session_state.connected:
            if st.session_state.db_connection:
                st.session_state.db_connection.close()

            st.session_state.db_connection = None
            st.session_state.nl_to_sql = None
            st.session_state.connected = False

            st.success("Disconnected from database")

        # -------- Connect --------
        else:
            conn = get_sqlserver_connection(connection_string)

            if conn:
                try:
                    st.session_state.db_connection = conn
                    st.session_state.connected = True

                    # Extract schema
                    schema_data = load_schemas()
                    # save_schema_to_json(schema_data, "schemas.json")

                    # Initialize FAISS
                    nl_to_sql = NLtoSQL()
                    nl_to_sql.initialize_schema(schema_data)
                    st.session_state.nl_to_sql = nl_to_sql

                    st.success("Connected to SQL Server and schema loaded successfully ✅")

                except Exception as e:
                    st.error(f"Error during initialization: {e}")
                    conn.close()
                    st.session_state.db_connection = None
                    st.session_state.connected = False
            else:
                st.error("Failed to connect to SQL Server")

    # Query input section
    if st.session_state.connected:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_query = st.chat_input("Ask a question about your database")

        if user_query:
            # Save user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_query}
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    # 1️⃣ Generate SQL
                    sql = st.session_state.nl_to_sql.process_natural_language(user_query)

                    print(sql)
                    # 2️⃣ Execute SQL
                    sql = sanitize_sql(sql)
                    print("Final SQL sent to DB:\n", sql)
                    
                    if not sql.lower().startswith("select"):
                        raise Exception("Only SELECT queries are allowed")

                    results = execute_sql(
                        st.session_state.db_connection,
                        sql
                    )

                    # 3️⃣ Rephrase result
                    answer = st.session_state.nl_to_sql.rephrase_result(
                        user_query,
                        sql,
                        results
                    )

                    # Optional: show SQL (debug)
                    with st.expander("🔍 Generated SQL"):
                        st.code(sql, language="sql")

                    # Show final answer
                    st.markdown(answer)

            # Save assistant response
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )


if __name__ == "__main__":
    main()
