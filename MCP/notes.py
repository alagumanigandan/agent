from mcp.server.fastmcp import FastMCP
import sqlite3
import os 
mcp=FastMCP("notes")

DB_PATH=os.path.abspath(os.path.join(os.getcwd(),"notes.db"))
def init_db():
    con=sqlite3.connect(DB_PATH)
    con.execute("""
            CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            note TEXT NOT NULL,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()
@mcp.tool()
async def save_note(topic: str, note: str, tags: str = ""):
    """ Save a research note into the SQLite database.
    Args:
        topic: Topic/category of the note.
        note:The actual research information.
        tags:Optional comma-separated tags.
    """
    con=sqlite3.connect(DB_PATH)
    no=con.execute(
        """
        INSERT INTO notes (topic, note, tags)
        VALUES (?, ?, ?)
        """,
        (topic, note, tags)
    )
    note_id = no.lastrowid
    con.commit()
    con.close()
    return f"Saved note #{note_id} under '{topic}'."
@mcp.tool()
async def  list_notes(topic: str = "")->str:
    """
    List research notes.
    
    Args:
        topic: Optional topic to filter notes.
               If empty, return all notes.
    """
    con= sqlite3.connect(DB_PATH)

    if topic:
        cursor = con.execute(
            """
            SELECT id, topic, note, tags, created_at
            FROM notes
            WHERE topic = ?
            ORDER BY created_at DESC
            """,
            (topic,)
        )

    else:
        cursor = con.execute(
            """
            SELECT id, topic, note, tags, created_at
            FROM notes
            ORDER BY created_at DESC
            """
        )

    rows = cursor.fetchall()

    con.close()

    if not rows:
        return "No notes found."

    return "\n".join(str(row) for row in rows)

@mcp.tool()
async def search_notes(keyword: str)->str:
    """  Search notes by keyword in topic, note, or tags.
    Args:
        keywoard:The word or phrase to search for
                 inside the topic, note content, and tags.
    """
    conn = sqlite3.connect(DB_PATH)

    search_term = f"%{keyword}%"

    cursor = conn.execute(
        """
        SELECT id, topic, note, tags, created_at
        FROM notes
        WHERE topic LIKE ?
           OR note LIKE ?
           OR tags LIKE ?
        ORDER BY created_at DESC
        """,
        (search_term, search_term, search_term)
    )

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return "No matching notes found."

    return "\n".join(str(row) for row in rows)
@mcp.tool()
async def tag_note(note_id: int, tag: str) -> str:
    """
    Add a tag to an existing research note.

    Args:
        note_id: The ID of the note to tag.
        tag: The label to add to the note.
    """
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.execute(
        """
        SELECT tags
        FROM notes
        WHERE id = ?
        """,
        (note_id,)
    )

    row = cursor.fetchone()

    if row is None:
        conn.close()
        return f"Note #{note_id} not found."

    existing_tags = row[0] or ""

    if existing_tags:
        new_tags = existing_tags + ", " + tag
    else:
        new_tags = tag
    conn.execute(
        """
        UPDATE notes
        SET tags = ?
        WHERE id = ?
        """,
        (new_tags, note_id)
    )

    conn.commit()
    conn.close()

    return f"Added tag '{tag}' to note #{note_id}."
if __name__=="__main__":
    init_db()
    mcp.run(transport="stdio")