import os
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
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
INPUT_DIR = Path("lectures")
OUTPUT_DIR = Path("embeddings")
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

def process_lecture_file(file_path, pbar=None):
    """Process a single lecture file and return units with embeddings"""
    global processed_units, failed_units
    
    try:
        logger.info(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lecture_id = file_path.stem
        units = data.get('units', [])
        
        if not units:
            logger.warning(f"No units found in {file_path}")
            return None
            
        logger.info(f"Found {len(units)} units in lecture {lecture_id}")
        
        # Process units in batches
        batch_num = 0
        total_batches = (len(units) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(units), BATCH_SIZE):
            batch_num += 1
            batch = units[i:i+BATCH_SIZE]
            
            # Create text for each unit in the batch
            texts = []
            for unit in batch:
                text = f"{unit.get('title', '')} {unit.get('statement', '')} {unit.get('comment', '')}"
                texts.append(text.strip())
            
            # Get embeddings for the batch
            logger.debug(f"Processing batch {batch_num}/{total_batches} for {lecture_id} with {len(batch)} units")
            embeddings = get_embeddings(texts)
            
            # Process embeddings
            for j, (unit, embedding) in enumerate(zip(batch, embeddings)):
                if embedding and 'embedding' in embedding:
                    unit['embedding'] = embedding['embedding']
                    processed_units += 1
                else:
                    failed_units += 1
                    logger.warning(f"Failed to get embedding for unit {unit.get('id', 'unknown')} in {lecture_id}")
                
                # Update progress bar if provided
                if pbar:
                    pbar.update(1)
            
            # Add delay between API calls
            if i + BATCH_SIZE < len(units):  # Don't sleep after the last batch
                time.sleep(DELAY_BETWEEN_REQUESTS)
        
        logger.info(f"Successfully processed {len(units)} units from {lecture_id}")
        return {
            'lecture_id': lecture_id,
            'units': units,
            'processed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return None

def save_embeddings(lecture_data, output_dir):
    """Save embeddings to individual JSON files"""
    if not lecture_data:
        return None
        
    try:
        lecture_id = lecture_data['lecture_id']
        output_file = output_dir / f"{lecture_id}_with_embeddings.json"
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(lecture_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved embeddings to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error saving embeddings for {lecture_data.get('lecture_id', 'unknown')}: {str(e)}")
        return None

def print_summary():
    """Print summary of the embedding generation process"""
    global total_units, processed_units, failed_units, api_calls, start_time
    
    end_time = time.time()
    duration = end_time - start_time if start_time else 0
    success_rate = (processed_units / total_units * 100) if total_units > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("EMBEDDING GENERATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total lecture files processed: {len(json_files)}")
    logger.info(f"Total units processed: {total_units}")
    logger.info(f"Successfully embedded units: {processed_units} ({success_rate:.2f}%)")
    logger.info(f"Failed units: {failed_units}")
    logger.info(f"Total API calls made: {api_calls}")
    logger.info(f"Time taken: {duration:.2f} seconds")
    logger.info(f"Average time per unit: {(duration/total_units):.2f}s" if total_units > 0 else "No units processed")
    logger.info("="*50 + "\n")

def main():
    global total_units, json_files, start_time
    
    start_time = time.time()
    logger.info("Starting embedding generation process")
    
    try:
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get all JSON files in the probability_theory directory
        json_files = sorted([f for f in INPUT_DIR.glob("*.json") if f.name != "schema.json"])
        
        if not json_files:
            logger.error(f"No JSON files found in {INPUT_DIR}")
            return
        
        # First pass: count total units for progress tracking
        total_units = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_units += len(data.get('units', []))
            except Exception as e:
                logger.error(f"Error counting units in {json_file}: {str(e)}")
        
        logger.info(f"Found {len(json_files)} lecture files with {total_units} total units to process")
        
        if total_units == 0:
            logger.error("No units found to process")
            return
        
        # Process files with a progress bar
        with tqdm(total=total_units, desc="Processing units", unit="unit") as pbar:
            for json_file in json_files:
                # Process the lecture file
                lecture_data = process_lecture_file(json_file, pbar)
                
                # Save the results
                save_embeddings(lecture_data, OUTPUT_DIR)
                
        # Print summary
        print_summary()
        
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        print_summary()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print_summary()
    finally:
        logger.info("Embedding generation process completed")

if __name__ == "__main__":
    main()
