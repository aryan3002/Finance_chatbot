import os, re, json, hashlib, time, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Union, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from pypdf import PdfReader
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)

# Paths
RAW_DIR = Path("data/raw")
DOWNLOAD_DIR = RAW_DIR / "downloads"
OUT_PATH = Path("data/processed/ingested.jsonl")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Regulatory source configuration
REGULATORY_SOURCES = {
    "SEC_RULES": {
        "base_url": "https://www.sec.gov/rules/",
        "patterns": [
            "final/", "proposed/", "interp/", "concept/"
        ],
        "file_types": [".pdf", ".htm", ".html"],
        "rate_limit": 10  # requests per second
    },
    "FINRA_RULES": {
        "base_url": "https://www.finra.org/rules-guidance/",
        "patterns": [
            "rulebooks/finra-rules/",
            "notices/",
            "guidance/"
        ],
        "file_types": [".pdf", ".htm", ".html"],
        "rate_limit": 5
    },
    "FEDERAL_RESERVE": {
        "base_url": "https://www.federalreserve.gov/",
        "patterns": [
            "supervisionreg/letters/",
            "regulations/",
            "guidance/"
        ],
        "file_types": [".pdf", ".htm", ".html"],
        "rate_limit": 5
    },
    "CFPB": {
        "base_url": "https://www.consumerfinance.gov/",
        "patterns": [
            "rules-policy/final-rules/",
            "compliance/compliance-guidance/",
            "policy-compliance/guidance/"
        ],
        "file_types": [".pdf", ".htm", ".html"],
        "rate_limit": 5
    },
    "FINCEN": {
        "base_url": "https://www.fincen.gov/",
        "patterns": [
            "resources/statutes-regulations/",
            "resources/guidance/",
            "resources/advisories/"
        ],
        "file_types": [".pdf", ".htm", ".html"],
        "rate_limit": 5
    }
}

def read_pdf_pymupdf(path: Path) -> str:
    """Read text from PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

def read_pdf_pypdf(path: Path) -> str:
    """Read text from PDF using PyPDF as fallback."""
    reader = PdfReader(str(path))
    texts = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(texts)

def clean_text(s: str) -> str:
    """Clean and normalize text."""
    s = re.sub(r"\s+", " ", s).strip()
    return s

class RegulatoryDocumentScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ComplianceBot/1.0 (Educational/Research Purpose; respectful of rate limits)'
        })
        self.downloaded_urls = set()
        self.error_urls = set()
    
    @sleep_and_retry
    @limits(calls=5, period=1)  # Default rate limit
    def _fetch_url(self, url: str, rate_limit: int = 5) -> requests.Response:
        """Fetch URL with rate limiting."""
        return self.session.get(url, timeout=30)
    
    def save_document(self, url: str, content: Union[str, bytes], is_binary: bool = False) -> Path:
        """Save downloaded document to disk."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        ext = Path(urlparse(url).path).suffix or '.txt'
        save_path = DOWNLOAD_DIR / f"{url_hash}{ext}"
        
        mode = 'wb' if is_binary else 'w'
        encoding = None if is_binary else 'utf-8'
        
        with open(save_path, mode, encoding=encoding) as f:
            f.write(content)
        
        return save_path
    
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from downloaded document."""
        try:
            if file_path.suffix.lower() == '.pdf':
                try:
                    return read_pdf_pymupdf(file_path)
                except:
                    return read_pdf_pypdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if file_path.suffix.lower() in ['.htm', '.html']:
                        return self.extract_html_content(content)
                    return content
        except Exception as e:
            logging.error(f"Failed to extract text from {file_path}: {e}")
            return None
    
    def extract_html_content(self, html_content: str) -> str:
        """Extract cleaned text content from HTML."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove non-content elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'iframe', 'aside']):
            element.decompose()
        
        # Find main content
        main_content = soup.find('main') or soup.find('article') or soup.find(['div', 'section'], 
            class_=re.compile(r'(content|main|body|text)'))
        content = main_content if main_content else soup
        
        # Extract text with structure
        sections = []
        
        # Get headings and their content
        for heading in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            section_text = [heading.get_text(strip=True)]
            current = heading.find_next_sibling()
            
            while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current.name in ['p', 'li', 'td']:
                    text = current.get_text(strip=True)
                    if len(text) > 20:  # Skip very short fragments
                        section_text.append(text)
                current = current.find_next_sibling()
            
            if len(section_text) > 1:
                sections.append('\n'.join(section_text))
        
        # Get any remaining paragraphs
        for p in content.find_all(['p', 'li']):
            if not any(p.find_parents(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
                text = p.get_text(strip=True)
                if len(text) > 20:
                    sections.append(text)
        
        return '\n\n'.join(sections)
    
    def find_document_links(self, url: str, file_types: List[str]) -> List[str]:
        """Find links to regulatory documents on a page."""
        try:
            response = self._fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                
                # Check if it's a document we want
                if any(full_url.lower().endswith(ft.lower()) for ft in file_types):
                    links.append(full_url)
                
            return links
        except Exception as e:
            logging.error(f"Failed to find links on {url}: {e}")
            return []
    
    def download_document(self, url: str) -> Optional[Dict]:
        """Download and process a single document."""
        if url in self.downloaded_urls or url in self.error_urls:
            return None
        
        try:
            response = self._fetch_url(url)
            response.raise_for_status()
            
            is_binary = 'application/pdf' in response.headers.get('Content-Type', '').lower()
            content = response.content if is_binary else response.text
            
            # Save document
            file_path = self.save_document(url, content, is_binary)
            
            # Extract text
            text = self.extract_text(file_path)
            if not text or len(text.strip()) < 200:
                self.error_urls.add(url)
                return None
            
            self.downloaded_urls.add(url)
            
            return {
                "id": hashlib.sha256(url.encode()).hexdigest()[:16],
                "source": url,
                "text": text,
                "metadata": {
                    "url": url,
                    "downloaded_at": datetime.now().isoformat(),
                    "file_type": file_path.suffix,
                    "file_path": str(file_path),
                    "source_type": "scraped"
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            self.error_urls.add(url)
            return None
    
    def scrape_source(self, source_name: str, config: Dict) -> List[Dict]:
        """Scrape documents from a regulatory source."""
        documents = []
        base_url = config['base_url']
        
        for pattern in config['patterns']:
            url = urljoin(base_url, pattern)
            logging.info(f"Scanning {url}")
            
            # Find document links
            doc_links = self.find_document_links(url, config['file_types'])
            
            # Download documents in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_url = {
                    executor.submit(self.download_document, link): link 
                    for link in doc_links
                }
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        doc = future.result()
                        if doc:
                            documents.append(doc)
                            logging.info(f"Downloaded: {url}")
                    except Exception as e:
                        logging.error(f"Failed to process {url}: {e}")
        
        return documents

def process_local_files() -> List[Dict]:
    """Process existing files in the data/raw directory."""
    docs = []
    
    for p in RAW_DIR.glob("**/*.*"):
        if p.is_file() and p.parent != DOWNLOAD_DIR:  # Skip downloaded files
            try:
                if p.suffix.lower() == '.pdf':
                    txt = read_pdf_pymupdf(p) or read_pdf_pypdf(p)
                elif p.suffix.lower() == '.txt':
                    txt = p.read_text(encoding='utf-8')
                else:
                    continue
                    
                txt = clean_text(txt)
                if len(txt) > 200:
                    docs.append({
                        "id": hashlib.sha256(str(p).encode()).hexdigest()[:16],
                        "source": str(p),
                        "text": txt,
                        "metadata": {
                            "file_type": p.suffix,
                            "file_path": str(p),
                            "source_type": "local"
                        }
                    })
                    logging.info(f"Processed local file: {p} ({len(txt)} chars)")
            except Exception as e:
                logging.error(f"Failed to process {p}: {e}")
    
    return docs

def add_supplementary_content() -> List[Dict]:
    """Add manual supplementary content."""
    docs = []
    
    supplementary_content = [
        {
            "title": "KYC Guidelines",
            "text": """
            Know Your Customer (KYC) Guidelines:
            1. Verify customer identity using government-issued ID
            2. Collect and verify proof of address
            3. Screen against sanctions lists
            4. Document source of funds for high-value transactions
            5. Conduct enhanced due diligence for high-risk customers
            6. Maintain records for at least 5 years
            7. Update customer information periodically
            8. Report suspicious activities to relevant authorities
            """
        },
        {
            "title": "AML Best Practices",
            "text": """
            Anti-Money Laundering Best Practices:
            1. Implement risk-based customer due diligence
            2. Monitor transactions for suspicious patterns
            3. Maintain comprehensive transaction records
            4. Train staff on AML procedures and red flags
            5. Conduct regular independent audits
            6. File SARs within required timeframes
            7. Document all investigation findings
            8. Update policies based on emerging risks
            """
        }
    ]
    
    for content in supplementary_content:
        doc_id = hashlib.sha256(content["title"].encode()).hexdigest()[:16]
        docs.append({
            "id": doc_id,
            "source": f"Guidelines - {content['title']}",
            "text": clean_text(content["text"]),
            "metadata": {
                "title": content["title"],
                "source_type": "supplementary"
            }
        })
        logging.info(f"Added supplementary content: {content['title']}")
    
    return docs

def main():
    """Main execution function."""
    logging.info("Starting regulatory document collection process...")
    
    all_docs = []
    
    # 1. Process local files
    local_docs = process_local_files()
    all_docs.extend(local_docs)
    logging.info(f"Processed {len(local_docs)} local files")
    
    # 2. Scrape regulatory sources
    scraper = RegulatoryDocumentScraper()
    
    for source_name, config in REGULATORY_SOURCES.items():
        logging.info(f"\nProcessing {source_name}...")
        try:
            scraped_docs = scraper.scrape_source(source_name, config)
            all_docs.extend(scraped_docs)
            logging.info(f"Downloaded {len(scraped_docs)} documents from {source_name}")
        except Exception as e:
            logging.error(f"Failed to process {source_name}: {e}")
    
    # 3. Add supplementary content
    supp_docs = add_supplementary_content()
    all_docs.extend(supp_docs)
    
    # Save all documents
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    logging.info(f"\nSaved {len(all_docs)} documents to {OUT_PATH}")
    
    # Print summary
    source_types = {}
    for doc in all_docs:
        source_type = doc.get("metadata", {}).get("source_type", "other")
        source_types[source_type] = source_types.get(source_type, 0) + 1
    
    logging.info("\nDocument source summary:")
    for source_type, count in source_types.items():
        logging.info(f"  {source_type}: {count} documents")

if __name__ == "__main__":
    main()
