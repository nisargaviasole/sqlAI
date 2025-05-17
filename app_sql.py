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

def generate_default_description(col_name, data_type, table_name, is_primary_key=False):
    data_type = data_type.upper()
    common_columns = {
        'ID': 'Unique identifier',
        'NAME': 'Name or title',
        'DATE': 'Date of the record',
        'TIME': 'Time of the record',
        'AMOUNT': 'Monetary or quantitative value',
        'NUMBER': 'Numeric identifier or count',
        'STATUS': 'Current state or condition',
        'DESCRIPTION': 'Detailed information or notes',
        'POLICY': 'Policy identifier or details',
        'PAYMENT': 'Payment amount or status',
        'CLAIM': 'Claim identifier or details',
        'INSURED': 'Insured party information',
    }
    col_name_upper = col_name.upper()
    prefix = "Primary key: " if is_primary_key else ""
    if col_name_upper in common_columns:
        return f"{prefix}{common_columns[col_name_upper]} for the {table_name} table"
    return f"{prefix}Stores {col_name} data of type {data_type} for the {table_name} table"

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
                    final_description = col_description if col_description else generate_default_description(col_name, data_type, table_name, is_pk)
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
        for table_info in schema_data:
            table_name = table_info["table_name"]
            columns = table_info["columns"]
            primary_keys = table_info.get("primary_keys", [])
            relationships = table_info.get("relationships", [])
            table_doc = (
                f"Table: {table_name}\n"
                f"Description: {table_info.get('description', 'No description available')}\n"
                f"Columns: {', '.join(columns)}\n"
                f"Primary keys: {', '.join(primary_keys) if primary_keys else 'None'}\n"
                f"Relationships: {len(relationships)} relationships to other tables"
            )
            documents.append(table_doc)
            for col in columns:
                col_info = table_info.get("column_details", {}).get(col, {})
                col_doc = (
                    f"Column: {col} in Table: {table_name}\n"
                    f"Data Type: {col_info.get('data_type', 'unknown')}\n"
                    f"Nullable: {'Yes' if col_info.get('is_nullable') == 'YES' else 'No'}\n"
                    f"Description: {col_info.get('description', 'No description available')}\n"
                    f"Example Usage: {table_name}.{col} stores {col_info.get('description', 'data')}"
                )
                documents.append(col_doc)
                if col.lower() in ['name', 'user', 'id', 'date', 'time', 'amount']:
                    alias_doc = (
                        f"Common Column: {col} (also known as {col.lower()}) in Table: {table_name}\n"
                        f"Data Type: {col_info.get('data_type', 'unknown')}\n"
                        f"Description: {col_info.get('description', 'No description available')}"
                    )
                    documents.append(alias_doc)
            for rel in relationships:
                rel_doc = (
                    f"Relationship: {table_name} to {rel['related_table']}\n"
                    f"Type: {rel['type']}\n"
                    f"Join Condition: {table_name}.{rel['from_column']} = {rel['related_table']}.{rel['to_column']}\n"
                    f"Description: Links records in {table_name} to {rel['related_table']}"
                )
                documents.append(rel_doc)
        self.vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        st.success(f"Loaded {len(documents)} schema documents into FAISS")

    def clean_sql_query(self, sql_query):
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:].strip()
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3].strip()
        sql_query = re.sub(r'(?i)\bselect\b', 'SELECT', sql_query)
        sql_query = re.sub(r'(?i)\bfrom\b', 'FROM', sql_query)
        sql_query = re.sub(r'(?i)\bwhere\b', 'WHERE', sql_query)
        sql_query = re.sub(r'(?i)\bjoin\b', 'JOIN', sql_query)
        sql_query = re.sub(r'(?i)\bgroup by\b', 'GROUP BY', sql_query)
        sql_query = re.sub(r'(?i)\border by\b', 'ORDER BY', sql_query)
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        sql_query = re.sub(r'(\S)(\(|\))', r'\1 \2', sql_query)
        sql_query = re.sub(r'(\(|\))(\S)', r'\1 \2', sql_query)
        return sql_query

    def validate_column_names(self, sql_query):
        if not self.schema_data:
            return sql_query
        schema_columns = set()
        schema_tables = set()
        for table in self.schema_data:
            schema_tables.add(table['table_name'])
            schema_columns.update(f"{table['table_name']}.{col}" for col in table['columns'])
        column_pattern = r'\[dbo\]\.\[(\w+)\]\.\[(\w+)\]'
        matches = re.finditer(column_pattern, sql_query, re.IGNORECASE)
        for match in matches:
            table_name, col_name = match.group(1), match.group(2)
            full_ref = f"{table_name}.{col_name}"
            if table_name not in schema_tables:
                return f"-- Error: Table [{table_name}] not found in schema"
            if full_ref not in schema_columns:
                for table in self.schema_data:
                    if table['table_name'].lower() == table_name.lower():
                        for col in table['columns']:
                            if col.lower() == col_name.lower():
                                correct_ref = f"[dbo].[{table['table_name']}].[{col}]"
                                sql_query = sql_query.replace(match.group(0), correct_ref)
                                break
                        else:
                            return f"-- Error: Column [{col_name}] not found in table [{table_name}]"
                        break
                else:
                    return f"-- Error: Table [{table_name}] not found in schema"
        return sql_query

    def process_natural_language(self, query):
        retrieved_docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = """You are an expert SQL query generator for Microsoft SQL Server. Your task is to convert natural language queries into precise, valid SQL queries based EXCLUSIVELY on the provided database schema.

IMPORTANT SAFETY RULE:
- If the user requests any query that would MODIFY data (INSERT, UPDATE, DELETE, ALTER, DROP, etc.),
  immediately respond with: "I cannot generate queries that modify the database."

Database Schema Context:
{context}

### STRICT REQUIREMENTS ###
1. OUTPUT FORMAT:
   - Generate ONLY the SQL query without any explanations, comments, or additional text
   - Format the query with proper line breaks for readability (not all in one line)
   - Use this exact formatting style:
     SELECT [dbo].[Table1].[Column1], [dbo].[Table2].[Column2]
     FROM [dbo].[Table1]
     INNER JOIN [dbo].[Table2] ON [dbo].[Table1].[ID] = [dbo].[Table2].[Table1ID]
     WHERE [dbo].[Table1].[Status] = 1
     ORDER BY [dbo].[Table2].[Date] DESC

2. IDENTIFIER FORMATTING:
   - Use square brackets [ ] for ALL identifiers
   - ALWAYS qualify table names with [dbo] schema (e.g., [dbo].[Customers])
   - For columns, use either:
     - [dbo].[TableName].[ColumnName] (preferred), or
     - [TableName].[ColumnName] if unambiguous

3. SCHEMA ADHERENCE RULES:
   - You MUST use EXACT table and column names from the schema (case-sensitive)
   - DOUBLE-CHECK that every table/column exists in the schema before using it
   - If the natural language term doesn't match any schema object, DO NOT include it
   - For ambiguous terms (like 'name', 'user', 'date'):
     - Check ALL tables in the schema for matching columns
     - Prefer columns that are primary/foreign keys
     - Prefer columns with descriptions matching the query intent
     - If still ambiguous, return "-- Ambiguous: [term] could match: [list possible columns]"

4. QUERY CONSTRUCTION RULES:
   - SELECT: Always specify exact columns (NEVER use SELECT *)
   - JOINS:
     - Use INNER JOIN unless LEFT JOIN is explicitly needed
     - Always use explicit ON clauses with proper join conditions
     - Join via primary/foreign key relationships when possible
   - FILTERING:
     - Include WHERE clauses for all filtering conditions
     - Use LOWER() for case-insensitive string comparisons
     - Use 1/0 for TRUE/FALSE in BIT fields
   - AGGREGATION:
     - Use GROUP BY when using aggregate functions
     - Include all non-aggregated columns in GROUP BY
   - ORDERING:
     - Use ORDER BY when sorting is implied
     - Prefer ordering by primary key when no specific sort is requested
   - LIMITING:
     - Use TOP instead of LIMIT (e.g., SELECT TOP 10 ...)
   - DATES:
     - Use SQL Server date functions (GETDATE(), DATEADD(), DATEDIFF())

5. ERROR HANDLING:
   - If the query is impossible given the schema:
     RETURN "-- Error: [clear reason why query can't be generated]"
   - If a required table/column is missing:
     RETURN "-- Missing: [specific table/column needed]"
   - If the natural language query is unclear:
     RETURN "-- Clarify: [what specific information is needed]"

6. SPECIAL CASES:
   - For user-related queries, check if joining with [dbo].[users] is appropriate
   - For date ranges, use proper date comparisons (>=, <=, BETWEEN)
   - For text searches, use LIKE with LOWER(): 
     WHERE LOWER([dbo].[Table].[Column]) LIKE LOWER('%search%')

### EXAMPLE OUTPUT ###
SELECT [dbo].[Orders].[OrderID], [dbo].[Customers].[CustomerName]
FROM [dbo].[Orders]
INNER JOIN [dbo].[Customers] ON [dbo].[Orders].[CustomerID] = [dbo].[Customers].[CustomerID]
WHERE [dbo].[Orders].[OrderDate] >= DATEADD(day, -30, GETDATE())
ORDER BY [dbo].[Orders].[OrderDate] DESC
""".format(context=context)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate a SQL query for: {query}"),
        ]

        try:
            response = llm.invoke(messages)
            sql_query = response.content.strip()
            sql_query = self.clean_sql_query(sql_query)
            sql_query = self.validate_column_names(sql_query)
            return sql_query
        except Exception as e:
            return f"-- Error generating SQL: {str(e)}"

def main():
    st.title("Natural Language to SQL Query Generator")
    st.write("Connect to the database and enter a natural language query to generate a SQL query.")
    if "db_connection" not in st.session_state:
        st.session_state.db_connection = None
    if "nl_to_sql" not in st.session_state:
        st.session_state.nl_to_sql = None
    if "connected" not in st.session_state:
        st.session_state.connected = False
    st.subheader("Database Connection")
    connect_button = st.button("Connect" if not st.session_state.connected else "Disconnect")
    if connect_button:
        if st.session_state.connected:
            if st.session_state.db_connection:
                st.session_state.db_connection.close()
            st.session_state.db_connection = None
            st.session_state.nl_to_sql = None
            st.session_state.connected = False
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
                        st.subheader("Generated SQL Query")
                        st.code(sql, language="sql")
                        if not sql.startswith(("-- Error:", "-- Ambiguous:", "-- Missing:", "-- Clarify:")):
                            if st.button("Execute Query"):
                                try:
                                    with st.session_state.db_connection.cursor() as cur:
                                        cur.execute(str(sql))
                                        if cur.description is not None:
                                            columns = [desc[0] for desc in cur.description]
                                            results = cur.fetchall()
                                            if results:
                                                df = pd.DataFrame(results, columns=columns)
                                                st.subheader("Query Results")
                                                st.dataframe(df)
                                            else:
                                                st.info("Query executed successfully but returned no results")
                                        else:
                                            rowcount = cur.rowcount
                                            st.session_state.db_connection.commit()
                                            st.success(f"Query executed successfully. Rows affected: {rowcount}")
                                except pyodbc.Error as e:
                                    st.error(f"Error executing query: {e}")
                                    st.session_state.db_connection.rollback()
                    except Exception as e:
                        st.error(f"Error generating SQL query: {e}")
            else:
                st.warning("Please enter a valid query.")
    else:
        st.warning("Please connect to the database before entering a query.")

if __name__ == "__main__":
    main()