"""
db_setup.py — SQLite schema setup. Run once (or just let auth.py auto-create
tables on first run -- see init_db() below, which is safe to call every time
your Streamlit app starts, since CREATE TABLE IF NOT EXISTS is a no-op if the
tables already exist).

No host/user/password needed -- SQLite is just a file on disk, not a server
you connect to. See conversation for the full explanation of why.
"""

import sqlite3

DB_PATH = r"C:\Users\dhwan\OneDrive\Documents\INTELLIFY\research_novelty.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # OFF by default in SQLite -- must set every connection
    conn.row_factory = sqlite3.Row            # lets you access columns by name, e.g. row["username"]
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            idea_description TEXT NOT NULL,
            matched_domains TEXT,
            research_goal TEXT,
            top_paper_id TEXT,
            top_similarity_score REAL,
            searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""PRAGMA table_info(search_history);""")
    conn.commit()
    conn.close()
    print(f"SQLite database ready at '{DB_PATH}'")


def log_search(user_id: int, idea_description: str, matched_domains: list,
               research_goal: str, top_paper_id: str, top_similarity_score: float):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO search_history
            (user_id, idea_description, matched_domains, research_goal, top_paper_id, top_similarity_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, idea_description, ",".join(matched_domains), research_goal, top_paper_id, top_similarity_score),
    )
    conn.commit()
    conn.close()


def get_user_history(user_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM search_history WHERE user_id = ? ORDER BY searched_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]   # row_factory makes each row dict-convertible


if __name__ == "__main__":
    init_db()