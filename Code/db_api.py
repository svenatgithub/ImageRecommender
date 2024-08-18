import sqlite3
import pickle
from config import DB_PATH

def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_tables(conn):
    """Create tables for embeddings, RGB histograms, and HSV histograms."""
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            image_id TEXT PRIMARY KEY,
            image_path TEXT,
            embedding BLOB NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rgb_histograms (
            image_id TEXT PRIMARY KEY,
            image_path TEXT,
            rgb_histogram BLOB NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hsv_histograms (
            image_id TEXT PRIMARY KEY,
            image_path TEXT,
            hsv_histogram BLOB NOT NULL
        )
    ''')

    conn.commit()

def insert_rgb_histogram(conn, image_id, image_path, rgb_histogram):
    """Insert RGB histogram for a specific image by image_id."""
    try:
        sql = '''INSERT OR REPLACE INTO rgb_histograms (image_id, image_path, rgb_histogram)
                 VALUES (?, ?, ?)'''
        cursor = conn.cursor()
        cursor.execute(sql, (image_id, image_path, pickle.dumps(rgb_histogram)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting RGB histogram: {e}")

def insert_hsv_histogram(conn, image_id, image_path, hsv_histogram):
    """Insert HSV histogram for a specific image by image_id."""
    try:
        sql = '''INSERT OR REPLACE INTO hsv_histograms (image_id, image_path, hsv_histogram)
                 VALUES (?, ?, ?)'''
        cursor = conn.cursor()
        cursor.execute(sql, (image_id, image_path, pickle.dumps(hsv_histogram)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting HSV histogram: {e}")

def insert_embedding(conn, image_id, image_path, embedding):
    """Insert embedding into the database."""
    try:
        sql = '''INSERT INTO embeddings (image_id, image_path, embedding)
                 VALUES (?, ?, ?)'''
        cursor = conn.cursor()
        cursor.execute(sql, (image_id, image_path, pickle.dumps(embedding)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting embedding for {image_id}: {e}")

def get_image_ids(conn):
    """Retrieve all image_ids from the metadata database."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT image_id FROM embeddings")
    image_ids = cursor.fetchall()
    return [row[0] for row in image_ids]

def get_histograms(conn, image_id):
    """Get the RGB and HSV histograms for a specific image by image_id."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT rgb_histogram FROM rgb_histograms WHERE image_id = ?", (image_id,))
        rgb_row = cursor.fetchone()

        cursor.execute("SELECT hsv_histogram FROM hsv_histograms WHERE image_id = ?", (image_id,))
        hsv_row = cursor.fetchone()

        if rgb_row and hsv_row:
            return pickle.loads(rgb_row[0]), pickle.loads(hsv_row[0])
        return None, None
    except sqlite3.Error as e:
        print(e)
        return None, None

def get_embedding(conn, image_id):
    """Get the embedding for a specific image by image_id."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        return None
    except sqlite3.Error as e:
        print(e)
        return None

def get_all_embeddings(conn):
    """Retrieve all embeddings from the database."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, embedding FROM embeddings")
        rows = cursor.fetchall()
        return {row[0]: pickle.loads(row[1]) for row in rows}
    except sqlite3.Error as e:
        print(e)
        return None

if __name__ == '__main__':
    conn = create_connection(DB_PATH)
    if conn is not None:
        create_tables(conn)
        conn.close()
