"""Script to download financial data from various sources."""
import os
import requests
from pathlib import Path
from typing import List, Dict
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data sources
FINANCIAL_SOURCES = {
    "sec_guidelines": {
        "url": "https://www.sec.gov/files/company-filing-manual.txt",
        "category": "regulatory"
    },
    "investment_basics": {
        "url": "https://www.investor.gov/introduction-investing/investing-basics",
        "category": "education"
    },
    "federal_reserve": {
        "url": "https://www.federalreserve.gov/publications/files/financial-stability-report.txt",
        "category": "market_analysis"
    },
    # Add more sources as needed
}

def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL and save to output_path."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        logger.info(f"Successfully downloaded: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False

def main():
    # Create necessary directories
    base_dir = Path("data/raw/finance")
    for category in set(source["category"] for source in FINANCIAL_SOURCES.values()):
        (base_dir / category).mkdir(parents=True, exist_ok=True)
    
    # Download files using thread pool
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for name, info in FINANCIAL_SOURCES.items():
            output_path = base_dir / info["category"] / f"{name}.txt"
            futures.append(
                executor.submit(download_file, info["url"], output_path)
            )
    
    # Log results
    successful = sum(1 for future in futures if future.result())
    logger.info(f"Downloaded {successful}/{len(FINANCIAL_SOURCES)} files")

if __name__ == "__main__":
    main()
