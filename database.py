import sqlite3
from typing import Dict, List
import json
import datetime

DB_NAME = "scheduler.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT,
            caption TEXT,
            image_path TEXT,
            schedule_time TEXT,
            status TEXT,   -- Draft, Pending, Approved, Published, Failed
            credentials TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_post(platform: str, caption: str, image_path: str, schedule_time: str, credentials: dict, status: str = "Approved") -> int:
    init_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO posts (platform, caption, image_path, schedule_time, status, credentials) VALUES (?, ?, ?, ?, ?, ?)",
        (platform, caption, image_path, schedule_time, status, json.dumps(credentials))
    )
    post_id = c.lastrowid
    conn.commit()
    conn.close()
    return post_id

def get_due_posts() -> List[Dict]:
    init_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Basic textual string comparison for ISO8601 dates works effectively
    c.execute("SELECT id, platform, caption, image_path, credentials FROM posts WHERE status = 'Approved' AND schedule_time <= ?", (now,))
    rows = c.fetchall()
    conn.close()
    
    return [
        {"id": r[0], "platform": r[1], "caption": r[2], "image_path": r[3], "credentials": json.loads(r[4])}
        for r in rows
    ]

def update_post_status(post_id: int, status: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE posts SET status = ? WHERE id = ?", (status, post_id))
    conn.commit()
    conn.close()
