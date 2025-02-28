import sqlite3

from aiohttp.hdrs import PRAGMA

from backend.src.pydantic_models import DocumentInfo

DBNAME = 'rag_ds_app.db'

def get_db_connection():
    conn = sqlite3.connect(DBNAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_document_table():

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS document_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        '''
    )

    conn.close()

    return True

def create_app_logs():
    conn = get_db_connection()

    conn.execute(''' 
        CREATE TABLE IF NOT EXISTS app_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_query TEXT,
        model_response TEXT,
        model TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')

    conn.close()


def insert_application_logs(session_id, user_query, model_response, model):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        'INSERT INTO app_logs (session_id, user_query, model_response, model) VALUES (?, ?, ?, ? )',
        (session_id, user_query, model_response, model)
    )
    conn.commit()
    conn.close()


def insert_document_record(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_data (filename) VALUES (?)', (filename,))
    # captures the id of the last inserted document
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return document_id

def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_data WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_data ORDER BY upload_timestamp desc')
    rows = cursor.fetchall()
    conn.close()

    documents = [DocumentInfo(id=row['id'],
                              filename=row['filename'],
                              upload_timestamp=row['upload_timestamp']) for row in rows]
    return documents

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT FROM app_logs WHERE session_id = ? ORDER BY created_at', (session_id))
    logs = cursor.fetchall()

    messages = []
    for row in logs:
        messages.extend([
            {"role": "human", "content": row["user_query"]},
            {"role": "ai", "content": row["model_response"]}
        ])
    conn.close()
    return messages

# init database
create_app_logs()
create_document_table()



