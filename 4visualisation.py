import sqlite3
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import logging
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Neo4jExporter:
    def __init__(self, sqlite_db: str = "knowledge_base.db"):
        # Load environment variables
        load_dotenv()
        
        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD must be set in .env file")
            
        # SQLite connection
        self.sqlite_conn = sqlite3.connect(sqlite_db)
        self.sqlite_conn.row_factory = sqlite3.Row
        
    def __enter__(self):
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'driver'):
            self.driver.close()
        self.sqlite_conn.close()
        
    def create_constraints(self):
        """Create necessary constraints in Neo4j"""
        with self.driver.session() as session:
            # Create constraints and indexes
            session.run("""
                CREATE CONSTRAINT knowledge_unit_id IF NOT EXISTS 
                FOR (n:KnowledgeUnit) REQUIRE n.id IS UNIQUE
            """)
            
            # Create tag constraint
            session.run("""
                CREATE CONSTRAINT tag_name IF NOT EXISTS 
                FOR (t:Tag) REQUIRE t.name IS UNIQUE
            """)
            
            # Create vector index if it doesn't exist
            session.run("""
                CREATE VECTOR INDEX knowledge_unit_embedding IF NOT EXISTS
                FOR (n:KnowledgeUnit) ON (n.embedding) 
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
    
    def _extract_source_tag(self, source: str) -> str:
        """Extract tag from source by taking text before the first '/'"""
        if not source:
            return "unknown"
        return source.split('/')[0].strip()
            
    def export_units(self):
        """Export all knowledge units from SQLite to Neo4j"""
        cursor = self.sqlite_conn.cursor()
        
        # Get all knowledge units
        cursor.execute("""
            SELECT id, title, statement, proof, tags, embedding, 
                   COALESCE(source, '') as source 
            FROM knowledge_units
        """)
        
        units = cursor.fetchall()
        logger.info(f"Found {len(units)} knowledge units to export")
        
        with self.driver.session() as session:
            for unit in units:
                try:
                    # Convert tags from JSON string to list if it's a string
                    tags = unit['tags']
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except json.JSONDecodeError:
                            tags = [tags] if tags else []
                    
                    # Extract source tag
                    source = unit['source'] if 'source' in unit else ''
                    source_tag = self._extract_source_tag(source)
                    
                    # Add source tag to tags if not already present
                    if source_tag and source_tag not in tags:
                        tags.append(source_tag)
                    
                    # Convert embedding from bytes to list if needed
                    embedding = unit['embedding']
                    if isinstance(embedding, bytes):
                        try:
                            embedding = list(embedding)
                        except Exception:
                            embedding = None
                    
                    # Create or update the node in Neo4j
                    session.execute_write(self._create_unit_node, unit, tags, embedding, source_tag)
                    
                    logger.debug(f"Exported unit {unit['id']}: {unit['title']}")
                    
                except Exception as e:
                    logger.error(f"Error exporting unit {unit.get('id')}: {e}")
                    
            logger.info(f"Successfully exported {len(units)} knowledge units")
    
    def _create_unit_node(self, tx, unit, tags, embedding, source_tag):
        """Helper method to create a unit node with its relationships"""
        # Create or update the knowledge unit
        tx.run("""
            MERGE (u:KnowledgeUnit {id: $id})
            SET u.title = $title,
                u.statement = $statement,
                u.proof = $proof,
                u.tags = $tags,
                u.embedding = $embedding,
                u.source = $source
        """, {
            'id': str(unit['id']),
            'title': unit['title'] or '',
            'statement': unit['statement'] or '',
            'proof': unit['proof'] or '',
            'tags': tags,
            'embedding': embedding,
            'source': unit['source'] if 'source' in unit else ''
        })
        
        # Create or update the tag node and create relationship
        if source_tag:
            tx.run("""
                MERGE (t:Tag {name: $tag_name})
                WITH t
                MATCH (u:KnowledgeUnit {id: $unit_id})
                MERGE (u)-[:TAGGED_WITH]->(t)
            """, {
                'tag_name': source_tag,
                'unit_id': str(unit['id'])
            })
            
    def export_connections(self):
        """Export all connections from SQLite to Neo4j"""
        cursor = self.sqlite_conn.cursor()
        
        # Get all connections
        cursor.execute("""
            SELECT source_id, target_id, relationship_type 
            FROM connections
        """)
        
        connections = cursor.fetchall()
        logger.info(f"Found {len(connections)} connections to export")
        
        with self.driver.session() as session:
            for conn in connections:
                try:
                    session.run("""
                        MATCH (source:KnowledgeUnit {id: $source_id})
                        MATCH (target:KnowledgeUnit {id: $target_id})
                        MERGE (source)-[r:PREREQUISITE]->(target)
                        SET r.relationship_type = $rel_type
                    """, {
                        'source_id': str(conn['source_id']),
                        'target_id': str(conn['target_id']),
                        'rel_type': conn['relationship_type'] or 'PREREQUISITE'
                    })
                    
                    logger.debug(f"Exported connection: {conn['source_id']} → {conn['target_id']}")
                    
                except Exception as e:
                    logger.error(f"Error exporting connection {conn['source_id']}→{conn['target_id']}: {e}")
                    
            logger.info(f"Successfully exported {len(connections)} connections")

def main():
    try:
        with Neo4jExporter() as exporter:
            # Create constraints and indexes
            logger.info("Creating Neo4j constraints and indexes...")
            exporter.create_constraints()
            
            # Export knowledge units
            logger.info("Exporting knowledge units...")
            exporter.export_units()
            
            # Export connections
            logger.info("Exporting connections...")
            exporter.export_connections()
            
            logger.info("Export completed successfully!")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
