import os
import json
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")
if not EDENAI_API_KEY:
    raise ValueError("Please set the EDENAI_API_KEY in your .env file")

# Constants
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
EMBEDDINGS_DIR = Path("embeddings")
API_URL = "https://api.edenai.run/v2/text/embeddings"
HEADERS = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "Content-Type": "application/json"
}
MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 10  # Number of texts to process in each API call
DELAY_BETWEEN_REQUESTS = 0.5  # Delay in seconds to avoid rate limiting

# Statistics
total_units = 0
processed_units = 0
failed_units = 0
api_calls = 0
start_time = None

def get_embeddings(texts, attempt=1, max_attempts=3):
    """Get embeddings for a list of texts using Eden AI API with retry logic"""
    global api_calls
    
    if not texts:
        return []
    
    data = {
        "providers": "openai",
        "texts": texts,
        "model": MODEL
    }
    
    try:
        logger.debug(f"Sending batch of {len(texts)} texts to Eden AI API")
        response = requests.post(API_URL, json=data, headers=HEADERS, timeout=30)
        response.raise_for_status()
        result = response.json()
        api_calls += 1
        
        if 'openai' not in result or 'items' not in result['openai']:
            raise ValueError("Unexpected API response format")
            
        return result['openai']['items']
        
    except requests.exceptions.RequestException as e:
        if attempt <= max_attempts:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Attempt {attempt}/{max_attempts} failed. Retrying in {wait_time}s... Error: {str(e)}")
            time.sleep(wait_time)
            return get_embeddings(texts, attempt + 1, max_attempts)
        else:
            logger.error(f"Failed after {max_attempts} attempts. Error: {str(e)}")
            return [None] * len(texts)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return [None] * len(texts)

def process_knowledge_file(file_path, pbar=None):
    """Process a single knowledge base file and return units with embeddings"""
    global processed_units, failed_units
    
    try:
        logger.info(f"Processing knowledge file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get relative path for logging and source tracking
        rel_path = file_path.relative_to(KNOWLEDGE_BASE_DIR)
        units = data.get('units', [])
        
        if not units:
            logger.warning(f"No knowledge units found in {rel_path}")
            return None
            
        logger.info(f"Found {len(units)} units in {rel_path}")
        
        # Process units in batches
        batch = []
        batch_indices = []
        
        for i, unit in enumerate(units):
            # Skip if already processed
            if 'embedding' in unit:
                continue
                
            # Create a text representation of the unit
            text_parts = []
            if unit.get('title'):
                text_parts.append(unit['title'])
            if unit.get('statement'):
                text_parts.append(unit['statement'])
            if unit.get('proof'):
                text_parts.append(unit['proof'])
                
            text = '\n'.join(text_parts).strip()
                
            if text:
                batch.append(text)
                batch_indices.append(i)
                
                # Process batch when it reaches BATCH_SIZE
                if len(batch) >= BATCH_SIZE:
                    embeddings = get_embeddings(batch)
                    for idx, emb in zip(batch_indices, embeddings):
                        if emb and 'embedding' in emb:
                            units[idx]['embedding'] = emb['embedding']
                            units[idx]['source'] = str(rel_path)
                            units[idx]['processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                            processed_units += 1
                            if pbar:
                                pbar.update(1)
                    
                    batch = []
                    batch_indices = []
                    
                    # Respect rate limiting
                    time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Process any remaining units in the last batch
        if batch:
            embeddings = get_embeddings(batch)
            for idx, emb in zip(batch_indices, embeddings):
                if emb and 'embedding' in emb:
                    units[idx]['embedding'] = emb['embedding']
                    units[idx]['source'] = str(rel_path)
                    units[idx]['processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    processed_units += 1
                    if pbar:
                        pbar.update(1)
        
        logger.info(f"Successfully processed {len(units)} units from {rel_path}")
        return units
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return None

def save_knowledge_embeddings(knowledge_data, output_dir):
    """Save knowledge units with embeddings to JSON files, preserving directory structure"""
    if not knowledge_data:
        return
    
    # Save each knowledge base file's embeddings
    for rel_path, data in knowledge_data.items():
        # Create output path preserving the directory structure
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"units": data}, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved embeddings to {output_file}")
    
    # Save a combined file with all embeddings
    combined_file = output_dir / "all_knowledge_embeddings.json"
    all_units = []
    for units in knowledge_data.values():
        all_units.extend(units)
    
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({"units": all_units}, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved combined knowledge embeddings to {combined_file}")
    return len(all_units)

def main():
    global total_units, start_time, processed_units, failed_units, api_calls
    
    # Reset counters
    total_units = 0
    processed_units = 0
    failed_units = 0
    api_calls = 0
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in the knowledge base directory
    json_files = list(KNOWLEDGE_BASE_DIR.glob('**/*.json'))
    
    if not json_files:
        logger.error(f"No JSON files found in {KNOWLEDGE_BASE_DIR}")
        return
    
    logger.info(f"Found {len(json_files)} knowledge base files to process")
    
    # First pass: count total units for progress tracking
    total_units = 0
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_units += len(data.get('units', []))
        except Exception as e:
            logger.error(f"Error counting units in {file_path}: {str(e)}")
    
    logger.info(f"Found {total_units} knowledge units to process")
    
    # Process each file and collect embeddings
    knowledge_data = {}
    
    # Initialize progress bar
    with tqdm(total=total_units, desc="Processing knowledge units") as pbar:
        for file_path in json_files:
            rel_path = file_path.relative_to(KNOWLEDGE_BASE_DIR)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process the knowledge units
                units = process_knowledge_file(file_path, pbar)
                if units:
                    knowledge_data[str(rel_path)] = units
                    
            except Exception as e:
                logger.error(f"Error processing {rel_path}: {str(e)}")
                failed_units += len(data.get('units', []))
    
    # Save embeddings to files
    total_saved = save_knowledge_embeddings(knowledge_data, EMBEDDINGS_DIR)
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info("\n=== Knowledge Embedding Generation Complete ===")
    logger.info(f"Total files processed: {len(json_files)}")
    logger.info(f"Total knowledge units: {total_units}")
    logger.info(f"Successfully processed: {processed_units}")
    logger.info(f"Failed to process: {failed_units}")
    logger.info(f"Total API calls made: {api_calls}")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per unit: {elapsed_time/max(1, processed_units):.2f} seconds")
    logger.info(f"Total units saved: {total_saved}")
    logger.info("Embedding generation process completed")

if __name__ == "__main__":
    main()
