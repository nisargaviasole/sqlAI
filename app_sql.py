import os
import json
import pyodbc
import streamlit as st
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import dotenv_values
from streamlit.watcher import local_sources_watcher
import re
import pandas as pd

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

# Set environment variables for cache and device
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Prevent CUDA interference

# Cache the embeddings object
@st.cache_resource
def load_embeddings():
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_folder = os.path.join(os.getcwd(), "hf_cache")
    model = SentenceTransformer(embedding_model_name, device="cpu", cache_folder=cache_folder)
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        cache_folder=cache_folder,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# Initialize embeddings once
embeddings = load_embeddings()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.0,
)

# Microsoft SQL Server connection
def get_db_connection():
    try:
        conn_str = (
            r"DRIVER={SQL Server};"
            r"SERVER=DESKTOP-GQD3PQ1\SQLEXPRESS;"
            r"DATABASE=chronoplot_DB2;"
            r"TrustServerCertificate=yes;"
        )
        conn = pyodbc.connect(conn_str)
        return conn
    except pyodbc.Error as e:
        sqlstate = e.args[0] if e.args else ""
        if sqlstate == "28000":
            st.error("Authentication failure: Windows account not authorized.")
        elif sqlstate == "IM002":
            st.error("ODBC driver not found. Ensure 'SQL Server' driver is installed.")
        else:
            st.error(f"Error connecting to Microsoft SQL Server: {e}")
        return None


def extract_mssql_schema(conn):
    if not conn:
        raise Exception("No database connection")
    schema_data = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo'")
            tables = [row[0] for row in cur.fetchall()]
            for table_name in tables:
                cur.execute(
                    """
                    SELECT CAST(value AS NVARCHAR(MAX)) AS description
                    FROM sys.extended_properties
                    WHERE major_id = OBJECT_ID('dbo.' + ?)
                    AND minor_id = 0
                    AND name = 'MS_Description'
                    """,
                    (table_name,),
                )
                table_description = cur.fetchone()[0] if cur.rowcount > 0 else ""
                cur.execute(
                    """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = 'dbo'
                    AND TABLE_NAME = ?
                    AND CONSTRAINT_NAME LIKE 'PK_%'
                    """,
                    (table_name,),
                )
                primary_keys = [row[0] for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT 
                        c.COLUMN_NAME,
                        c.DATA_TYPE,
                        c.IS_NULLABLE,
                        c.CHARACTER_MAXIMUM_LENGTH,
                        CAST(ep.value AS NVARCHAR(MAX)) AS column_description
                    FROM INFORMATION_SCHEMA.COLUMNS c
                    LEFT JOIN sys.columns sc
                        ON sc.object_id = OBJECT_ID('dbo.' + ?)
                        AND sc.name = c.COLUMN_NAME
                    LEFT JOIN sys.extended_properties ep
                        ON ep.major_id = sc.object_id
                        AND ep.minor_id = sc.column_id
                        AND ep.name = 'MS_Description'
                    WHERE c.TABLE_SCHEMA = 'dbo' 
                    AND c.TABLE_NAME = ?
                    ORDER BY c.ORDINAL_POSITION
                    """,
                    (table_name, table_name),
                )
                columns = []
                column_details = {}
                for col_name, data_type, is_nullable, char_max_len, col_description in cur.fetchall():
                    columns.append(col_name)
                    is_pk = col_name in primary_keys
                    final_description = col_description
                    column_details[col_name] = {
                        "data_type": data_type.upper(),
                        "is_nullable": is_nullable,
                        "max_length": char_max_len,
                        "description": final_description,
                    }
                cur.execute(
                    """
                    SELECT 
                        fk.name AS constraint_name,
                        c1.name AS from_column,
                        t2.name AS related_table,
                        c2.name AS to_column
                    FROM sys.foreign_keys fk
                    INNER JOIN sys.tables t1 ON fk.parent_object_id = t1.object_id
                    INNER JOIN sys.tables t2 ON fk.referenced_object_id = t2.object_id
                    INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                    INNER JOIN sys.columns c1 ON fkc.parent_object_id = c1.object_id AND fkc.parent_column_id = c1.column_id
                    INNER JOIN sys.columns c2 ON fkc.referenced_object_id = c2.object_id AND fkc.referenced_column_id = c2.column_id
                    WHERE t1.name = ?
                    AND SCHEMA_NAME(t1.schema_id) = 'dbo'
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
                table_info = {
                    "table_name": table_name,
                    "description": table_description,
                    "columns": columns,
                    "column_details": column_details,
                    "primary_keys": primary_keys,
                    "relationships": relationships,
                }
                schema_data.append(table_info)
    finally:
        pass
    return schema_data

def save_schema_to_json(schema_data, filename="sql_schemas.json"):
    try:
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "w") as f:
            json.dump(schema_data, f, indent=4)
        st.success(f"Schema saved to {filename}")
    except Exception as e:
        st.error(f"Error saving schema to {filename}: {e}")

def load_schema_from_json(filename="sql_schemas.json"):
    try:
        with open(filename, "r") as f:
            schema_data = json.load(f)
        st.success(f"Schema loaded from {filename}")
        return schema_data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Error loading schema from {filename}: {e}")
        return None

class NLtoSQL:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.schema_data = None

    def initialize_schema(self, schema_data):
        self.schema_data = schema_data
        documents = []
        self.table_lookup = {}  # Add this for faster table lookups
        
        for table_info in schema_data:
            table_name = table_info["table_name"]
            self.table_lookup[table_name] = table_info  # Store for quick access

        for table_info in schema_data:
            table_name = table_info["table_name"]
            columns = table_info["columns"]
            primary_keys = table_info.get("primary_keys", [])
            relationships = table_info.get("relationships", [])
            column_details = table_info.get("column_details", {})

            # Group full table schema into one document
            column_lines = []
            for col in columns:
                col_info = column_details.get(col, {})
                dtype = col_info.get("data_type", "unknown")
                nullable = "YES" if col_info.get("is_nullable") == "YES" else "NO"
                description = col_info.get("description", "No description")
                column_lines.append(
                    f"  - {col} ({dtype}, Nullable: {nullable}): {description}"
                )

            relationship_lines = []
            for rel in relationships:
                # Get details about the related table
                related_table_info = self.table_lookup.get(rel['related_table'], {})
                related_pk = related_table_info.get('primary_keys', ['id'])[0]
                
                rel_line = (
                    f"{table_name}.{rel['from_column']} → {rel['related_table']}."
                    f"{rel.get('to_column', related_pk)} ({rel['type']})"
                )
                relationship_lines.append(rel_line)

            table_doc = f"""Table: {table_name}
    Primary Keys: {', '.join(primary_keys) if primary_keys else 'None'}

    Columns:
    {chr(10).join(column_lines)}

    Relationships:
    {chr(10).join(relationship_lines) if relationship_lines else 'None'}
    """
            documents.append(table_doc.strip())

        # Index documents in FAISS
        self.vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        st.success(f"Loaded {len(documents)} schema documents into FAISS.")


    def process_natural_language(self, query):
        # Step 1: Get initial relevant tables from FAISS
        retrieved_docs = self.retriever.get_relevant_documents(query)  # Changed from invoke()
        
        # Step 2: Extract all mentioned tables from retrieved docs
        mentioned_tables = set()
        for doc in retrieved_docs:
            content = doc.page_content  # Now properly accessing page_content
            if content.startswith("Table:"):
                table_name = content.split("\n")[0].replace("Table:", "").strip()
                mentioned_tables.add(table_name)
        
        # Step 3: Find all relationally connected tables
        related_tables = set()
        for table in mentioned_tables:
            # Find direct relationships (tables this table connects to)
            for table_info in self.schema_data:
                if table_info["table_name"] == table:
                    for rel in table_info.get("relationships", []):
                        related_tables.add(rel["related_table"])
            
            # Find inverse relationships (tables that connect to this table)
            for table_info in self.schema_data:
                for rel in table_info.get("relationships", []):
                    if rel["related_table"] == table:
                        related_tables.add(table_info["table_name"])
        
        # Step 4: Get schema documents for all related tables
        all_tables = mentioned_tables.union(related_tables)
        expanded_docs = []
        
        for table in all_tables:
            # Find the table's document in the original schema data
            for table_info in self.schema_data:
                if table_info["table_name"] == table:
                    # Recreate the document format you used in initialize_schema
                    table_doc = self._create_table_document(table_info)
                    expanded_docs.append(table_doc)
                    break
        
        # Step 5: Use all this context for SQL generation
        context = "\n\n".join(expanded_docs)
        print(context)
        system_prompt = """You are a SQL query generator for Microsoft SQL Server (T-SQL). Your task is to convert natural language 
            queries into valid T-SQL queries based strictly on the database schema provided.

            Here is the relevant database schema information:
            {context}

            **Instructions**:
            - Return only the SQL query without any explanation or formatting.
            - Use only column names and table names exactly as they appear in the schema. Do not assume or hallucinate any field or table.
            - Convert Boolean values to integers: use 1 for true and 0 for false.
            - Use square brackets for all table names and column names (e.g., [users], [order_date]).
            - Ensure the query is valid T-SQL syntax for SQL Server.
            - For queries involving specific users (e.g., fetching orders for a user by name), use INNER JOIN to join the [users] table with the relevant table using the relationship between [users].[id] and <table>.[user_id].
            - Use LEFT OUTER JOIN only when the query explicitly requires including non-matching rows from the primary table.
            - Do not use subqueries (e.g., WHERE [user_id] = (SELECT [id] FROM [users] ...)) unless absolutely necessary.
            - Use LOWER([column]) LIKE '%value%' for case-insensitive string matching.
            - If a required column or table is not present in the schema, return a comment like: -- Additional information needed: <details>.
            - In the query if a new column made Give it a Proper name Which can define the column.
            - JUST RETURN QUERY ONLY DO NOT ADD QUOTS TO IT.
            """.format(context=context)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate a SQL query for: {query}"),
        ]

        try:
            response = llm.invoke(messages)
            sql_query = response.content.strip()
            return sql_query
        except Exception as e:
            return f"-- Error generating SQL: {str(e)}"

    def _create_table_document(self, table_info):
        """Helper to recreate the same document format used in initialize_schema"""
        table_name = table_info["table_name"]
        columns = table_info["columns"]
        primary_keys = table_info.get("primary_keys", [])
        relationships = table_info.get("relationships", [])
        column_details = table_info.get("column_details", {})

        column_lines = []
        for col in columns:
            col_info = column_details.get(col, {})
            dtype = col_info.get("data_type", "unknown")
            nullable = "YES" if col_info.get("is_nullable") == "YES" else "NO"
            description = col_info.get("description", "No description")
            column_lines.append(
                f"  - {col} ({dtype}, Nullable: {nullable}): {description}"
            )

        relationship_lines = []
        for rel in relationships:
            rel_line = (
                f"{table_name}.{rel['from_column']} → {rel['related_table']}."
                f"{rel['to_column']} ({rel['type']})"
            )
            relationship_lines.append(rel_line)

        table_doc = f"""Table: {table_name}
    Primary Keys: {', '.join(primary_keys) if primary_keys else 'None'}

    Columns:
    {chr(10).join(column_lines)}

    Relationships:
    {chr(10).join(relationship_lines) if relationship_lines else 'None'}
    """
        return table_doc.strip()


def validate_sql_query(sql: str) -> bool:
    """Basic validation of SQL query syntax"""
    if not sql or sql.strip().startswith("--"):
        return False
    # Check for common SQL operations
    sql_upper = sql.upper()
    has_select = "SELECT" in sql_upper
    has_from = "FROM" in sql_upper
    # For non-SELECT queries
    has_operation = any(op in sql_upper for op in ["INSERT", "UPDATE", "DELETE"])
    return (has_select and has_from) or has_operation

def main():
    st.title("Natural Language to SQL Query Generator")
    st.write("Connect to the database and enter a natural language query to generate a SQL query.")
    if "db_connection" not in st.session_state:
        st.session_state.db_connection = None
    if "nl_to_sql" not in st.session_state:
        st.session_state.nl_to_sql = None
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "generated_sql" not in st.session_state:
        st.session_state.generated_sql = None  # Initialize generated_sql

    st.subheader("Database Connection")
    connect_button = st.button("Connect" if not st.session_state.connected else "Disconnect")
    if connect_button:
        if st.session_state.connected:
            if st.session_state.db_connection:
                st.session_state.db_connection.close()
            st.session_state.db_connection = None
            st.session_state.nl_to_sql = None
            st.session_state.connected = False
            st.session_state.generated_sql = None  # Clear query on disconnect
            st.success("Disconnected from database")
        else:
            conn = get_db_connection()
            if conn:
                st.session_state.db_connection = conn
                st.session_state.connected = True
                try:
                    schema_file = "sql_schemas.json"
                    schema_data = extract_mssql_schema(conn)
                    save_schema_to_json(schema_data, schema_file)
                    nl_to_sql = NLtoSQL()
                    nl_to_sql.initialize_schema(schema_data)
                    st.session_state.nl_to_sql = nl_to_sql
                    st.success("Connected to database chronoplot_DB2")
                except Exception as e:
                    st.error(f"Error extracting schema: {e}")
                    schema_data = load_schema_from_json(schema_file)
                    if schema_data:
                        nl_to_sql = NLtoSQL()
                        nl_to_sql.initialize_schema(schema_data)
                        st.session_state.nl_to_sql = nl_to_sql
                        st.success("Loaded existing schema from sql_schemas.json")
                    else:
                        conn.close()
                        st.session_state.db_connection = None
                        st.session_state.connected = False
                        st.error("Failed to initialize NLtoSQL.")
            else:
                st.error("Failed to connect to the database")

    if st.session_state.connected:
        st.subheader("Query Input")
        query = st.text_input(
            "Enter your natural language query:",
            placeholder="e.g., Count incidents per location for safeguarding",
        )
        if st.button("Generate SQL Query"):
            if query.strip():
                if st.session_state.nl_to_sql is None:
                    st.error("NLtoSQL is not initialized. Please reconnect to the database.")
                else:
                    try:
                        with st.spinner("Generating SQL query..."):
                            sql = st.session_state.nl_to_sql.process_natural_language(query)
                            st.session_state.generated_sql = sql
                    except Exception as e:
                        st.error(f"Error generating SQL query: {e}")

        # Always display the generated SQL query if it exists
        if st.session_state.generated_sql:
            st.subheader("Generated SQL Query")
            st.code(st.session_state.generated_sql, language="sql")

        st.subheader("Execute Generated Query")
        if st.button("Execute Query", key="execute_query_button"):
            if not st.session_state.generated_sql:
                st.warning("No SQL query generated. Please generate a query first.")
                return
            if not st.session_state.db_connection:
                st.error("Database connection is not active. Please reconnect.")
                return
            sql = st.session_state.generated_sql
            if sql.startswith(("-- Error:", "-- Ambiguous:", "-- Missing:", "-- Clarify:")):
                st.warning(f"Cannot execute: {sql}")
                return
            is_valid = validate_sql_query(sql)
            if is_valid:
                try:
                    with st.session_state.db_connection.cursor() as cur:
                        cur.execute(str(sql))
                        if cur.description:
                            columns = [desc[0] for desc in cur.description]
                            data = cur.fetchall()
                            data = [tuple(row) for row in data]
                            df = pd.DataFrame(data, columns=columns)
                            st.table(df)
                        else:
                            st.success("Query executed successfully (no results returned).")
                            st.session_state.db_connection.commit()
                except pyodbc.Error as e:
                    error_msg = str(e)
                    if "Invalid object name" in error_msg:
                        st.error(f"Table not found: {error_msg}")
                    elif "Invalid column name" in error_msg:
                        st.error(f"Column not found: {error_msg}")
                    else:
                        st.error(f"Database error: {error_msg}")
                    st.session_state.db_connection.rollback()
            else:
                st.warning("Generated query appears invalid - please review before executing")
    else:
        st.warning("Please connect to the database before entering a query.")

if __name__ == "__main__":
    main()