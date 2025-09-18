"""Script to extract and process financial concepts from downloaded documents."""
import os
from pathlib import Path
import json
import logging
from typing import Dict, List
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define key financial concepts to extract
FINANCIAL_CONCEPTS = {
    "investment": [
        "stocks", "bonds", "mutual funds", "ETFs", "portfolio",
        "diversification", "risk management", "asset allocation"
    ],
    "trading": [
        "market orders", "limit orders", "day trading", "margin",
        "short selling", "options", "futures"
    ],
    "banking": [
        "deposits", "loans", "interest rates", "credit", "mortgages",
        "savings accounts", "checking accounts"
    ],
    "corporate_finance": [
        "balance sheet", "income statement", "cash flow", "ratios",
        "valuation", "mergers", "acquisitions"
    ],
    "risk_management": [
        "hedging", "insurance", "derivatives", "risk assessment",
        "compliance", "internal controls"
    ]
}

def extract_concepts(text: str) -> Dict[str, List[str]]:
    """Extract relevant financial concepts from text."""
    found_concepts = {category: [] for category in FINANCIAL_CONCEPTS}
    
    for category, terms in FINANCIAL_CONCEPTS.items():
        for term in terms:
            # Look for the term and surrounding context
            pattern = f"[^.]*{term}[^.]*\\."
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            # Store found concepts with context
            for match in matches:
                found_concepts[category].append(match.group(0).strip())
    
    return found_concepts

def process_file(file_path: Path) -> Dict:
    """Process a single file and extract financial concepts."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract concepts
        concepts = extract_concepts(content)
        
        # Create metadata
        metadata = {
            "source": file_path.name,
            "category": file_path.parent.name,
            "concepts_found": {k: len(v) for k, v in concepts.items()}
        }
        
        return {
            "metadata": metadata,
            "concepts": concepts,
            "raw_text": content
        }
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    base_dir = Path("data/raw/finance")
    processed_dir = Path("data/processed/finance")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    for file_path in base_dir.rglob("*.txt"):
        logger.info(f"Processing: {file_path}")
        
        result = process_file(file_path)
        if result:
            # Save processed data
            output_path = processed_dir / f"{file_path.stem}_processed.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved processed data to: {output_path}")

if __name__ == "__main__":
    main()
