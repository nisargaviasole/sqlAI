import os
import json
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

# Initialize embedding model
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name, cache_folder=os.path.join(os.getcwd(), "hf_cache",model_kwargs={"device": "cpu"})
)   

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
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


# PostgreSQL connection
def get_db_connection(pg_host, pg_database, pg_user, pg_password, pg_port):
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


def extract_postgres_schema(conn):
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


def save_schema_to_json(schema_data, filename="schemas.json"):
    """
    Save the schema data to a JSON file.
    """
    try:
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
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = f"""You are a SQL query generator for PostgreSQL. Your task is to convert natural language 
        queries into valid PostgreSQL queries based on the database schema provided.
        
        Here is the relevant database schema information:
        {context}
        
        **Instructions**:
        - Generate only the SQL query without any explanations.
        - Convert Boolean value in to lowercase every time. For example if its TRUE then convert it in true.
        - Use double quotes on table names and all column names.
        - Ensure the query is valid PostgreSQL syntax.
        - For queries involving specific users (e.g., fetching orders for a user by name), use **INNER JOIN** to join the `users` table with the relevant table (e.g., `orders`) using the relationship between `users.id` and `<table>.user_id`.
        - Use **LEFT OUTER JOIN** only when the query explicitly requires including non-matching rows from the primary table.
        - Avoid using subqueries (e.g., `WHERE user_id = (SELECT id FROM users...)`) unless JOINs cannot express the query.
        - If the user's query references a user by name (e.g., 'nisarg'), assume the name is in the `users.name` column and join with the `users` table.
        - Use PostgreSQL-specific features (e.g., `ILIKE` for case-insensitive search) where appropriate.
        - If the schema information is insufficient or the query is ambiguous, return a comment (e.g., `-- Additional information needed: <details>`).
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


def main():
    st.title("Natural Language to SQL Query Generator")
    st.write(
        "Select a database host from the dropdown to connect and enter a natural language query to generate a SQL query."
    )

    # Initialize session state variables
    if "db_connection" not in st.session_state:
        st.session_state.db_connection = None
    if "nl_to_sql" not in st.session_state:
        st.session_state.nl_to_sql = None
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "selected_host" not in st.session_state:
        st.session_state.selected_host = None

    # Database connection form
    st.subheader("Database Connection")
    credentials = load_credentials()
    host_display_map = {
        f"database = {cred['Database']}, hostname = {cred['Host']}": cred['Host']
        for cred in credentials
    } if credentials else {"No hosts available": None}

    host_options = list(host_display_map.keys())

    with st.form(key="db_form"):
        selected_display = st.selectbox(
            "Select Host",
            host_options,
            index=(
                host_options.index(st.session_state.selected_host)
                if st.session_state.selected_host in host_options
                else 0
            ),
        )
        
        # Get the actual hostname
        selected_host = host_display_map[selected_display]
        connect_button = st.form_submit_button(
            "Connect" if not st.session_state.connected else "Disconnect"
        )

    # Handle connect/disconnect
    if connect_button:
        if st.session_state.connected:
            # Disconnect
            if st.session_state.db_connection:
                st.session_state.db_connection.close()
            st.session_state.db_connection = None
            st.session_state.nl_to_sql = None
            st.session_state.connected = False
            st.session_state.selected_host = None
            st.success("Disconnected from database")
        else:
            # Connect
            if selected_host != "No hosts available":
                # Find the credentials for the selected host
                selected_cred = next(
                    (cred for cred in credentials if cred["Host"] == selected_host),
                    None,
                )
                if selected_cred:
                    conn = get_db_connection(
                        selected_cred["Host"],
                        selected_cred["Database"],
                        selected_cred["Username"],
                        selected_cred["Database_Pass"],
                        selected_cred["Port"],
                    )
                    if conn:
                        st.session_state.db_connection = conn
                        st.session_state.connected = True
                        st.session_state.selected_host = selected_host
                        # Extract and initialize schema
                        try:
                            schema_file = f"schemas.json"
                            schema_data = extract_postgres_schema(conn)
                            save_schema_to_json(schema_data, schema_file)
                            nl_to_sql = NLtoSQL()
                            nl_to_sql.initialize_schema(schema_data)
                            st.session_state.nl_to_sql = nl_to_sql
                            st.success(
                                f"Connected to database {selected_cred['Database']} on {selected_host}"
                            )
                        except Exception as e:
                            st.error(f"Error extracting schema: {e}")
                            conn.close()
                            st.session_state.db_connection = None
                            st.session_state.connected = False
                            st.session_state.selected_host = None
                    else:
                        st.error("Failed to connect to the database")
                else:
                    st.error("Selected host not found in credentials")
            else:
                st.error("No hosts available to connect")

    # Query input section
    if st.session_state.connected:
        st.subheader("Query Input")
        query = st.text_input(
            "Enter your natural language query:",
            placeholder="e.g., List policy numbers where the payment is counted",
        )

        if st.button("Generate SQL Query"):
            if query.strip():
                try:
                    with st.spinner("Generating SQL query..."):
                        sql = st.session_state.nl_to_sql.process_natural_language(query)
                    st.subheader("Generated SQL Query")
                    st.code(sql, language="sql")
                except Exception as e:
                    st.error(f"Error generating SQL query: {e}")
            else:
                st.warning("Please enter a valid query.")
    else:
        st.warning(
            "Please select a host and connect to a database before entering a query."
        )


if __name__ == "__main__":
    main()
