import sqlite3
import json
import struct
import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseHandler:
    """
    A handler class for database operations related to knowledge units and their connections.
    Provides simplified interfaces for adding, updating, and querying knowledge units and their relationships.
    """
    
    def __init__(self, db_path: Union[str, Path] = "knowledge_base.db"):
        """
        Initialize the database handler.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        
        # Check if vector extension is available
        self.vector_extension_available = False
        
        # Try to import sqlite-vss
        try:
            import sqlite_vss
            
            # Try to load the extension
            try:
                # First try the standard way
                self.conn.enable_load_extension(True)
                self.conn.load_extension('vector')
            except (AttributeError, sqlite3.OperationalError):
                # If that fails, try using sqlite_vss.load()
                sqlite_vss.load(self.conn)
            
            # Test if vector operations work
            cursor = self.conn.cursor()
            try:
                cursor.execute("SELECT vss_version()")
                version = cursor.fetchone()[0]
                logger.info(f"Vector extension loaded successfully (v{version})")
                self.vector_extension_available = True
            except (sqlite3.OperationalError, IndexError):
                # If vss_version() doesn't exist, try a simple vector operation
                try:
                    cursor.execute("SELECT vector_to_json('[1.0, 2.0, 3.0]')")
                    logger.info("Vector extension loaded successfully")
                    self.vector_extension_available = True
                except sqlite3.OperationalError as e:
                    logger.warning(f"Vector operations not available: {str(e)}")
                    
        except ImportError:
            logger.warning("sqlite-vss package not installed")
        except Exception as e:
            logger.warning(f"Could not initialize vector extension: {str(e)}")
            
        if not self.vector_extension_available:
            logger.info("Falling back to FTS-only mode")
    
    def __del__(self):
        """Close the database connection when the object is destroyed"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def close(self):
        """Explicitly close the database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
    
    def add_knowledge_unit(self, unit_data: Dict[str, Any]) -> int:
        """
        Add a new knowledge unit to the database.
        
        Args:
            unit_data: Dictionary containing unit data with keys matching the database columns
                     (original_id, lecture_id, kind, title, statement, proof, tags, source)
                     
        Returns:
            int: The ID of the newly inserted unit
        """
        # Make a copy of the data to avoid modifying the original
        data = unit_data.copy()
        
        # Convert tags to JSON string if it's a list/dict
        if 'tags' in data and data['tags'] is not None:
            if not isinstance(data['tags'], str):
                data['tags'] = json.dumps(data['tags'], ensure_ascii=False)
        else:
            data['tags'] = '[]'  # Default to empty JSON array
            
        # Ensure all values are strings or None
        for key in data:
            if data[key] is None:
                data[key] = ''
            elif not isinstance(data[key], (str, int, float)):
                data[key] = str(data[key])
        
        # Prepare the SQL query
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        query = f"""
        INSERT INTO knowledge_units ({columns})
        VALUES ({placeholders})
        """
        
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, list(data.values()))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                # If the unit already exists, return its ID
                cursor.execute(
                    "SELECT id FROM knowledge_units WHERE original_id = ? AND lecture_id = ?",
                    (data.get('original_id'), data.get('lecture_id'))
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
            logger.error(f"Error adding knowledge unit: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding knowledge unit: {str(e)}")
            raise
    
    def add_connection(self, source_id: str, target_id: str, relationship_type: str = "related", weight: float = 1.0) -> int:
        """
        Add a connection between two knowledge units.
        
        Args:
            source_id: The ID of the source unit
            target_id: The ID of the target unit
            relationship_type: Type of relationship (e.g., "prerequisite", "related")
            weight: Strength/weight of the connection (default: 1.0)
            
        Returns:
            int: The ID of the newly created connection
        """
        query = """
        INSERT INTO connections (source_id, target_id, relationship_type, weight)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_id, target_id, relationship_type) 
        DO UPDATE SET weight = excluded.weight
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (source_id, target_id, relationship_type, weight))
        self.conn.commit()
        return cursor.lastrowid
    
    def add_embedding(self, unit_id: int, embedding: List[float]) -> bool:
        """
        Add or update a vector embedding for a knowledge unit.
        
        Args:
            unit_id: The ID of the knowledge unit
            embedding: The embedding vector as a list of floats
            
        Returns:
            bool: True if successful, False if vector extension is not available
        """
        if not self.vector_extension_available:
            return False
            
        try:
            cursor = self.conn.cursor()
            
            # Check if the embedding column exists in the knowledge_units table
            cursor.execute("PRAGMA table_info(knowledge_units)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add the embedding column if it doesn't exist
            if 'embedding' not in columns:
                cursor.execute('''
                ALTER TABLE knowledge_units 
                ADD COLUMN embedding BLOB
                ''')
            
            # Convert the embedding to a JSON string for storage
            import json
            embedding_json = json.dumps(embedding)
            
            # Update the knowledge_units table with the embedding
            cursor.execute(
                """
                UPDATE knowledge_units 
                SET embedding = ?
                WHERE id = ?
                """,
                (embedding_json, unit_id)
            )
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error adding embedding for unit {unit_id}: {str(e)}", exc_info=True)
            self.conn.rollback()
            return False
    
    def get_unit_by_id(self, unit_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a knowledge unit by its ID.
        
        Args:
            unit_id: The ID of the unit to retrieve
            
        Returns:
            Optional[Dict]: The unit data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM knowledge_units WHERE id = ?", (unit_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return dict(row)
    
    def get_connections(self, unit_id: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve all connections for a given unit.
        
        Args:
            unit_id: The ID of the unit
            relationship_type: Optional filter for relationship type
            
        Returns:
            List[Dict]: List of connection dictionaries
        """
        query = """
        SELECT c.*, 
               ku1.title as source_title, 
               ku2.title as target_title
        FROM connections c
        LEFT JOIN knowledge_units ku1 ON c.source_id = ku1.id
        LEFT JOIN knowledge_units ku2 ON c.target_id = ku2.id
        WHERE c.source_id = ? OR c.target_id = ?
        """
        
        params = [unit_id, unit_id]
        
        if relationship_type:
            query += " AND c.relationship_type = ?"
            params.append(relationship_type)
            
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def find_similar_units(self, embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find units with similar embeddings using vector similarity search.
        
        Args:
            embedding: The query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List[Dict]: List of similar units with their similarity scores
        """
        if not self.vector_extension_available:
            return []
            
        # Convert the embedding to a format suitable for the vector extension
        embedding_blob = sqlite3.Binary(bytes(bytearray(struct.pack(f'<{len(embedding)}f', *embedding))))
        
        query = """
        SELECT ku.*, 
               vector_distance(kv.vector_embedding, ?) as distance
        FROM knowledge_units ku
        JOIN knowledge_vectors kv ON ku.id = kv.embedding_id
        ORDER BY distance ASC
        LIMIT ?
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (embedding_blob, limit))
        
        return [dict(row) for row in cursor.fetchall()]
        
    def search_units_by_text(self, search_expression: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for knowledge units using FTS5 full-text search.
        
        Args:
            search_expression: The search expression (e.g., 'теорема AND катетов')
            limit: Maximum number of results to return (default: 5)
            
        Returns:
            List[Dict]: List of top 5 matching units with their data (excluding embeddings)
        """
        cursor = self.conn.cursor()
        
        # Use the FTS5 table to perform the search with correct syntax
        search_query = """
        SELECT 
            ku.id,
            ku.original_id,
            ku.lecture_id,
            ku.kind,
            ku.title,
            ku.statement,
            ku.proof,
            ku.tags,
            ku.source,
            kf.rank
        FROM knowledge_units ku
        JOIN (
            SELECT rowid, rank
            FROM knowledge_fts
            WHERE knowledge_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ) kf ON ku.id = kf.rowid
        ORDER BY kf.rank
        """
        
        try:
            cursor.execute(search_query, (search_expression, limit))
            results = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Convert tags from JSON string back to list if needed
                if 'tags' in result and isinstance(result['tags'], str):
                    try:
                        result['tags'] = json.loads(result['tags'])
                    except (json.JSONDecodeError, TypeError):
                        result['tags'] = []
                # Ensure we don't include embeddings
                if 'embedding' in result:
                    del result['embedding']
                results.append(result)
            return results
                
        except sqlite3.OperationalError as e:
            logger.error(f"Error performing full-text search: {str(e)}")
            return []

    def find_similar_by_embedding(self, embedding, limit=5):
        """
        Find similar knowledge units using vector similarity search.
        
        Args:
            embedding: The embedding vector to compare against
            limit: Maximum number of results to return
                    
        Returns:
            List of similar knowledge units with their similarity scores
        """
        if not self.vector_extension_available:
            logger.warning("Vector extension not available, returning empty results")
            return []
                    
        try:
            # Convert embedding to a list if it's not already
            if not isinstance(embedding, list):
                if isinstance(embedding, str):
                    # If it's a JSON string, parse it
                    embedding = json.loads(embedding)
                elif hasattr(embedding, 'tolist'):
                    # If it's a numpy array, convert to list
                    embedding = embedding.tolist()
                else:
                    # Otherwise, try to convert to list directly
                    embedding = list(embedding)
            
            cursor = self.conn.cursor()
            
            # Get all knowledge units with embeddings
            cursor.execute("""
                SELECT id, title, statement, embedding
                FROM knowledge_units
                WHERE embedding IS NOT NULL
            """)
            
            results = []
            
            # Calculate cosine similarity between the query embedding and each unit's embedding
            for row in cursor.fetchall():
                try:
                    # Parse the stored embedding
                    stored_embedding = json.loads(row['embedding'])
                    
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(embedding, stored_embedding))
                    norm_a = sum(a * a for a in embedding) ** 0.5
                    norm_b = sum(b * b for b in stored_embedding) ** 0.5
                    
                    if norm_a == 0 or norm_b == 0:
                        similarity = 0.0
                    else:
                        similarity = dot_product / (norm_a * norm_b)
                    
                    # Add to results
                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'statement': row['statement'],
                        'similarity': similarity
                    })
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error processing embedding for unit {row['id']}: {str(e)}")
                    continue
            
            # Sort by similarity in descending order and take top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:limit]
            
            return results
                        
        except Exception as e:
            logger.error(f"Error in find_similar_by_embedding: {str(e)}", exc_info=True)
            return []
