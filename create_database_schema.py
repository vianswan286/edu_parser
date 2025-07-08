import sqlite3
import os

class DatabaseSchema:
    def __init__(self, db_path: str = "knowledge_base.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        
        # Try to load the vector extension
        try:
            self.conn.load_extension('vector')
            print("SQLite vector extension loaded successfully.")
        except sqlite3.OperationalError:
            print("Warning: SQLite vector extension not found. Some features may not work.")
    
    def create_tables(self):
        """Create all necessary tables in the database."""
        cursor = self.conn.cursor()
        
        # Drop existing tables if they exist
        cursor.execute('DROP TABLE IF EXISTS connections')
        cursor.execute('DROP TABLE IF EXISTS knowledge_units')
        cursor.execute('DROP TABLE IF EXISTS knowledge_vectors')
        
        # Create knowledge_units table
        cursor.execute('''
        CREATE TABLE knowledge_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT,  -- Original ID from JSON
            lecture_id TEXT,
            kind TEXT,
            title TEXT,
            statement TEXT,
            proof TEXT,
            tags TEXT,  -- Stored as JSON array
            source TEXT,
            embedding BLOB,  -- Stored as binary data
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(original_id, lecture_id)  -- Ensure no duplicates
        )
        ''')
        
        # Create connections table
        cursor.execute('''
        CREATE TABLE connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT,
            target_id TEXT,
            relationship_type TEXT,
            weight FLOAT DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES knowledge_units (id),
            FOREIGN KEY (target_id) REFERENCES knowledge_units (id),
            UNIQUE (source_id, target_id, relationship_type)
        )
        ''')
        
        # Create vector index if extension is available
        try:
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vectors 
            USING vector(
                embedding_id INTEGER PRIMARY KEY,
                vector_embedding,
                FOREIGN KEY (embedding_id) REFERENCES knowledge_units(id)
            )
            ''')
            print("Vector table created successfully.")
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not create vector virtual table: {e}")
        
        # Create FTS5 virtual table for full-text search
        try:
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts 
            USING fts5(
                id UNINDEXED,  -- Store but don't index the ID
                original_id UNINDEXED,
                lecture_id,
                kind,
                title,
                statement,
                proof,
                tags,
                source UNINDEXED,
                content='knowledge_units',  -- External content table
                content_rowid='rowid'       -- Rowid of the content table
            )
            ''')
            
            # Create triggers to keep the FTS index in sync with the main table
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_units
            BEGIN
                INSERT INTO knowledge_fts (
                    rowid, id, original_id, lecture_id, kind, title, 
                    statement, proof, tags, source
                ) VALUES (
                    new.rowid, new.id, new.original_id, new.lecture_id, new.kind, 
                    new.title, new.statement, new.proof, new.tags, new.source
                );
            END;
            ''')
            
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_units
            BEGIN
                DELETE FROM knowledge_fts WHERE rowid = old.rowid;
            END;
            ''')
            
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_units
            BEGIN
                UPDATE knowledge_fts SET 
                    id = new.id,
                    original_id = new.original_id,
                    lecture_id = new.lecture_id,
                    kind = new.kind,
                    title = new.title,
                    statement = new.statement,
                    proof = new.proof,
                    tags = new.tags,
                    source = new.source
                WHERE rowid = old.rowid;
            END;
            ''')
            
            print("FTS5 virtual table and triggers created successfully.")
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not create FTS5 virtual table: {e}")
        
        # Populate the FTS table with existing data
        try:
            cursor.execute('''
            INSERT INTO knowledge_fts (
                rowid, id, original_id, lecture_id, kind, title, 
                statement, proof, tags, source
            )
            SELECT 
                rowid, id, original_id, lecture_id, kind, title, 
                statement, proof, tags, source 
            FROM knowledge_units;
            ''')
            self.conn.commit()
            print("FTS5 table populated with existing data.")
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not populate FTS5 table: {e}")
        
        self.conn.commit()
        print(f"Database schema created successfully at {self.db_path}")
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


def main():
    # Initialize the database schema
    db_path = "knowledge_base.db"
    
    # Delete existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Create and set up the database
    db = DatabaseSchema(db_path)
    db.create_tables()
    db.close()
    
    print("\nDatabase setup complete!")


if __name__ == "__main__":
    main()
