"""
Database for Semantic Video Search System
Creates SQLite database with tables for videos, search history, and feedback
"""
import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_search.db")
DATASET_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "msr_vtt_dataset.csv")


def create_database():
    """Create all database tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table 1: Videos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            title TEXT,
            category TEXT,
            caption TEXT,
            link TEXT,
            duration_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 2: Search History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            search_type TEXT DEFAULT 'text',
            language TEXT DEFAULT 'en',
            results_count INTEGER DEFAULT 0,
            avg_score REAL DEFAULT 0.0,
            top_k INTEGER DEFAULT 5,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 3: User Feedback
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            query TEXT,
            feedback_type TEXT CHECK(feedback_type IN ('up', 'down')),
            score REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 4: System Stats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_searches INTEGER DEFAULT 0,
            total_feedbacks INTEGER DEFAULT 0,
            avg_search_time REAL DEFAULT 0.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("Database created successfully: " + DB_PATH)


def load_videos_from_csv():
    """Load video data from CSV into database"""
    if not os.path.exists(DATASET_CSV):
        print("CSV not found: " + DATASET_CSV)
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_csv(DATASET_CSV, encoding="utf-8")

    count = 0
    for _, row in df.iterrows():
        try:
            link = str(row.get("Link", "")).strip()
            video_id = link.split("/")[-1].replace(".mp4", "") if link else f"video{count}"
            title = str(row.get("Title", "")).strip()
            caption = str(row.get("Caption", "")).strip()
            category = title.split("_")[0] if "_" in title else "unknown"

            conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, title, category, caption, link) VALUES (?, ?, ?, ?, ?)",
                (video_id, title, category, caption, link)
            )
            count += 1
        except Exception as e:
            continue

    conn.commit()
    conn.close()
    print(f"Loaded {count} videos into database")


def save_search(query, search_type="text", language="en", results_count=0, avg_score=0.0, top_k=5):
    """Save a search query to history"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO search_history (query, search_type, language, results_count, avg_score, top_k) VALUES (?, ?, ?, ?, ?, ?)",
        (query, search_type, language, results_count, avg_score, top_k)
    )
    conn.commit()
    conn.close()


def save_feedback(video_id, query, feedback=None, feedback_type=None, score=0.0):
    """Save user feedback — يقبل feedback أو feedback_type"""
    fb = feedback or feedback_type or "up"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO feedback (video_id, query, feedback_type, score) VALUES (?, ?, ?, ?)",
        (video_id, query, fb, score)
    )
    conn.commit()
    conn.close()


def get_search_history(limit=50):
    """Get recent search history"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT query, search_type, language, results_count, avg_score, timestamp FROM search_history ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_feedback_stats():
    """Get feedback statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN feedback_type = 'up' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN feedback_type = 'down' THEN 1 ELSE 0 END) as negative
        FROM feedback
    """)
    result = cursor.fetchone()
    conn.close()
    return {"total": result[0], "positive": result[1], "negative": result[2]}

def get_all_feedback():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_video_count():
    """Get total number of videos in database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM videos")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_popular_searches(limit=10):
    """Get most popular search queries"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT query, COUNT(*) as count, AVG(avg_score) as avg_score
        FROM search_history
        GROUP BY query
        ORDER BY count DESC
        LIMIT ?
    """, (limit,))
    results = cursor.fetchall()
    conn.close()
    return results


def get_database_stats():
    """Get overall database statistics"""
    conn = sqlite3.connect(DB_PATH)
    videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    searches = conn.execute("SELECT COUNT(*) FROM search_history").fetchone()[0]
    feedbacks = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    categories = conn.execute("SELECT COUNT(DISTINCT category) FROM videos").fetchone()[0]
    conn.close()
    return {
        "total_videos": videos,
        "total_searches": searches,
        "total_feedbacks": feedbacks,
        "total_categories": categories
    }


# Run this file to create database and load data
if __name__ == "__main__":
    print("=" * 50)
    print("Setting up Video Search Database")
    print("=" * 50)

    # Step 1: Create tables
    create_database()

    # Step 2: Load videos from CSV
    load_videos_from_csv()

    # Step 3: Show stats
    stats = get_database_stats()
    print(f"\nDatabase Stats:")
    print(f"  Videos:     {stats['total_videos']}")
    print(f"  Categories: {stats['total_categories']}")
    print(f"  Searches:   {stats['total_searches']}")
    print(f"  Feedbacks:  {stats['total_feedbacks']}")
    print("\nDone!")