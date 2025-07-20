#!/usr/bin/env python3
import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lecture_parsing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeType(str, Enum):
    DEFINITION = "definition"
    AXIOM = "axiom"
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    FACT = "fact"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    NOTE = "note"
    REMARK = "remark"
    IDEA = "idea"
    NOTATION = "notation"
    DESIGNATION = "designation"

@dataclass
class KnowledgeUnit:
    id: str
    kind: str
    title: str = ""
    statement: str = ""
    proof: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""

class LatexParser:
    def __init__(self):
        self.patterns = {
            KnowledgeType.DEFINITION: [
                re.compile(r'\\begin{definition}(?:\[(.*?)\])?\s*(.*?)\\end{definition}', re.DOTALL),
                re.compile(r'\\defn\{(.*?)\}\s*\{(.*?)\}', re.DOTALL)
            ],
            KnowledgeType.THEOREM: [
                re.compile(r'\\begin{theorem}(?:\[(.*?)\])?\s*(.*?)\\end{theorem}', re.DOTALL),
                re.compile(r'\\thm\{(.*?)\}\s*\{(.*?)\}', re.DOTALL)
            ],
            # Add more patterns for other knowledge types
        }
    
    def parse_file(self, file_path: Path) -> List[Dict]:
        """Parse a LaTeX file and extract knowledge units."""
        try:
            content = file_path.read_text(encoding='utf-8')
            units = []
            
            # Extract document class and packages for context
            doc_class = re.search(r'\\documentclass\[(.*?)\]\{(.*?)\}', content)
            
            # Process each knowledge type
            for ktype, patterns in self.patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(content):
                        title = match.group(1).strip() if len(match.groups()) > 1 and match.group(1) else ""
                        statement = match.group(2).strip() if len(match.groups()) > 1 else match.group(0).strip()
                        
                        # Clean up the statement
                        statement = self._clean_latex(statement)
                        
                        unit = KnowledgeUnit(
                            id=f"{ktype[0].upper()}{len(units) + 1}",
                            kind=ktype.value,
                            title=title,
                            statement=statement,
                            source=str(file_path.relative_to("material"))
                        )
                        units.append(asdict(unit))
            
            return {"units": units}
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return {"units": []}
    
    def _clean_latex(self, text: str) -> str:
        """Clean up LaTeX commands and special characters."""
        # Remove comments
        text = re.sub(r'\s*%.*\n', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def process_directory(input_dir: Path, output_dir: Path):
    """Process all LaTeX files in the input directory and save them as JSON."""
    parser = LatexParser()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each LaTeX file
    for latex_file in input_dir.glob('**/*.tex'):
        logger.info(f"Processing {latex_file}...")
        
        # Parse the LaTeX file
        knowledge_base = parser.parse_file(latex_file)
        
        if knowledge_base["units"]:
            # Create output path
            relative_path = latex_file.relative_to(input_dir)
            json_path = output_dir / f"{relative_path.with_suffix('')}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
            
            logger.info(f"  Extracted {len(knowledge_base['units'])} knowledge units to {json_path}")
        else:
            logger.warning(f"  No knowledge units found in {latex_file}")

def main():
    # Configuration
    input_dir = Path("material")
    output_dir = Path("knowledge_base")
    
    # Process all LaTeX files
    process_directory(input_dir, output_dir)
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
