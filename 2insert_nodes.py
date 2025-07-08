#!/usr/bin/env python3
"""
Script to insert knowledge units from the embeddings folder into the database.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

# Import our database handler
from database_handler import DatabaseHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insert_nodes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDINGS_DIR = Path("embeddings")
DB_PATH = "knowledge_base.db"

class KnowledgeInserter:
    """Handles the insertion of knowledge units and their connections into the database."""
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize the knowledge inserter with a database connection."""
        self.db = DatabaseHandler(db_path)
        self.stats = {
            'total_files': 0,
            'processed_units': 0,
            'processed_connections': 0,
            'skipped_units': 0,
            'errors': 0
        }
    
    def process_embeddings_dir(self, embeddings_dir: Path) -> None:
        """Process all JSON files in the embeddings directory."""
        if not embeddings_dir.exists() or not embeddings_dir.is_dir():
            logger.error(f"Embeddings directory not found: {embeddings_dir}")
            return
        
        # Get all JSON files in the embeddings directory
        json_files = list(embeddings_dir.glob("*.json"))
        self.stats['total_files'] = len(json_files)
        
        if not json_files:
            logger.warning(f"No JSON files found in {embeddings_dir}")
            return
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        for json_file in tqdm(json_files, desc="Processing files"):
            self.process_embedding_file(json_file)
    
    def process_embedding_file(self, file_path: Path) -> None:
        """Process a single embedding JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lecture_id = data.get('lecture_id', file_path.stem)
            units = data.get('units', [])
            
            if not units:
                logger.warning(f"No units found in {file_path}")
                return
            
            logger.info(f"Processing {len(units)} units from {lecture_id}")
            
            # Process each unit in the file
            for unit in tqdm(units, desc=f"Processing {lecture_id}", leave=False):
                self.process_unit(unit, lecture_id)
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
    
    def process_unit(self, unit: Dict[str, Any], lecture_id: str) -> Optional[int]:
        """Process a single knowledge unit and insert it into the database."""
        try:
            # Prepare unit data for insertion (excluding embedding)
            unit_data = {
                'original_id': str(unit.get('id', '')),  # Ensure ID is a string
                'lecture_id': lecture_id,
                'kind': unit.get('kind', 'unknown'),
                'title': unit.get('title', ''),
                'statement': unit.get('statement', ''),
                'proof': unit.get('proof', ''),
                'tags': unit.get('tags', []),
                'source': unit.get('source', '')
            }
            
            # Insert the unit into the database
            unit_id = self.db.add_knowledge_unit(unit_data)
            
            # If we have an embedding, store it separately
            if 'embedding' in unit and unit['embedding'] is not None:
                # Convert the embedding to a list of floats if it's not already
                embedding = unit['embedding']
                if not isinstance(embedding, list):
                    embedding = list(embedding)  # Convert numpy array or other iterables to list
                self.db.add_embedding(unit_id, embedding)
            
            self.stats['processed_units'] += 1
            return unit_id
            
        except Exception as e:
            self.stats['errors'] += 1
            self.stats['skipped_units'] += 1
            logger.error(f"Error processing unit {unit.get('id', 'unknown')}: {str(e)}")
            return None
    
    def print_stats(self) -> None:
        """Print statistics about the insertion process."""
        logger.info("\n" + "=" * 50)
        logger.info("INSERTION STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Knowledge units inserted/updated: {self.stats['processed_units']}")
        logger.info(f"Skipped units: {self.stats['skipped_units']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info("=" * 50 + "\n")

def main():
    """Main function to run the knowledge insertion process."""
    logger.info("Starting knowledge unit insertion process")
    start_time = datetime.now()
    
    try:
        # Initialize and run the knowledge inserter
        inserter = KnowledgeInserter()
        inserter.process_embeddings_dir(EMBEDDINGS_DIR)
        
        # Print final statistics
        inserter.print_stats()
        
        # Calculate and log total time taken
        duration = datetime.now() - start_time
        logger.info(f"Process completed in {duration}")
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
