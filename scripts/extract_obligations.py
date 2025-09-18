# scripts/extract_obligations.py
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional
import joblib
from sentence_transformers import SentenceTransformer

from prompts import EXTRACTION_PROMPT, SUMMARY_PROMPT

# Paths
DOCSTORE_PATH = Path("artifacts/docstore.pkl")
CLASSIFIER_PATH = Path("artifacts/topic_classifier.joblib")
OUTPUT_PATH = Path("data/processed/obligations.jsonl")

# Patterns for regulatory extraction
OBLIGATION_PATTERNS = [
    r"(?:must|shall|required to|obligated to)\s+([^.]+)",
    r"(?:within|no later than)\s+(\d+)\s+(?:days?|months?|years?)",
    r"(?:retain|maintain|keep)\s+(?:records?|documents?)\s+(?:for)\s+([^.]+)",
    r"(?:report|notify|inform)\s+(?:to)?\s*([^.]+)",
    r"(?:prohibited from|may not|cannot)\s+([^.]+)",
]

DEADLINE_PATTERNS = [
    r"(\d+)\s*(?:calendar\s+)?days?",
    r"(\d+)\s*(?:business\s+)?days?",
    r"(\d+)\s*months?",
    r"(\d+)\s*years?",
    r"(?:annually|quarterly|monthly|weekly|daily)",
]

def extract_obligations_regex(text: str) -> Dict:
    """Extract compliance obligations using regex patterns."""
    obligations = []
    deadlines = []
    retention_periods = []
    
    # Extract obligations
    for pattern in OBLIGATION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        obligations.extend([m.strip() for m in matches if isinstance(m, str)])
    
    # Extract deadlines
    for pattern in DEADLINE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        deadlines.extend([m for m in matches if m])
    
    # Extract retention periods
    retention_pattern = r"retain.*?for\s+([^,.]+(?:years?|months?|days?))"
    ret_matches = re.findall(retention_pattern, text, re.IGNORECASE)
    retention_periods.extend(ret_matches)
    
    return {
        "obligations": list(set(obligations[:5])),  # Top 5 unique
        "deadlines": list(set(deadlines[:3])),
        "retention_periods": list(set(retention_periods[:2])),
    }

def classify_compliance_topic(text: str, classifier=None) -> str:
    """Classify text into compliance category."""
    if classifier is None:
        # Fallback to keyword-based classification
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["kyc", "know your customer", "identity verification"]):
            return "KYC"
        elif any(kw in text_lower for kw in ["money laundering", "aml", "suspicious activity"]):
            return "AML"
        elif any(kw in text_lower for kw in ["sanction", "ofac", "embargo"]):
            return "Sanctions"
        elif any(kw in text_lower for kw in ["sar", "suspicious activity report"]):
            return "SAR"
        elif any(kw in text_lower for kw in ["pci", "payment card", "cardholder data"]):
            return "PCI"
        elif any(kw in text_lower for kw in ["privacy", "gdpr", "data protection", "personal information"]):
            return "Privacy"
        else:
            return "General"
    
    try:
        return classifier.predict([text])[0]
    except:
        return "General"

def extract_section_references(text: str) -> List[str]:
    """Extract section/regulation references."""
    patterns = [
        r"(?:Section|Sec\.?|§)\s*(\d+(?:\.\d+)*)",
        r"(?:Article|Art\.?)\s*(\d+(?:\.\d+)*)",
        r"(?:Regulation|Reg\.?)\s*([A-Z]+(?:-\d+)?)",
        r"(?:Rule|R\.)\s*(\d+(?:\.\d+)*)",
        r"(\d{2}\s+(?:CFR|USC)\s+§?\s*\d+(?:\.\d+)*)",  # US regulations
    ]
    
    references = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        references.extend(matches)
    
    return list(set(references[:5]))  # Top 5 unique

def analyze_risk_level(text: str) -> str:
    """Determine risk level based on keywords."""
    high_risk_keywords = ["criminal", "penalty", "fine", "imprisonment", "violation", "prohibited", "illegal"]
    medium_risk_keywords = ["must", "shall", "required", "mandatory", "comply"]
    
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in high_risk_keywords):
        return "HIGH"
    elif any(kw in text_lower for kw in medium_risk_keywords):
        return "MEDIUM"
    else:
        return "LOW"

def main():
    """Extract obligations from indexed documents."""
    
    # Load docstore
    if not DOCSTORE_PATH.exists():
        print("Error: Run chunk_and_index.py first!")
        return
    
    with open(DOCSTORE_PATH, "rb") as f:
        docstore = pickle.load(f)
    
    # Try to load classifier
    classifier = None
    if CLASSIFIER_PATH.exists():
        try:
            classifier = joblib.load(CLASSIFIER_PATH)
            print("Loaded compliance classifier")
        except:
            print("Using keyword-based classification")
    
    # Process each chunk
    obligations_data = []
    
    for idx, (text, metadata) in enumerate(zip(docstore["texts"], docstore["metadatas"])):
        if len(text) < 100:  # Skip very short chunks
            continue
        
        # Extract structured data
        extracted = extract_obligations_regex(text)
        topic = classify_compliance_topic(text, classifier)
        sections = extract_section_references(text)
        risk_level = analyze_risk_level(text)
        
        # Create obligation record
        obligation = {
            "chunk_id": idx,
            "source": metadata.get("source", "unknown"),
            "topic": topic,
            "risk_level": risk_level,
            "obligations": extracted["obligations"],
            "deadlines": extracted["deadlines"],
            "retention_periods": extracted["retention_periods"],
            "section_references": sections,
            "text_snippet": text[:500],  # First 500 chars for context
        }
        
        obligations_data.append(obligation)
        
        if idx % 50 == 0:
            print(f"Processed {idx} chunks...")
    
    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for obj in obligations_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"\nExtracted obligations from {len(obligations_data)} chunks")
    print(f"Saved to: {OUTPUT_PATH}")
    
    # Print summary statistics
    topics = [o["topic"] for o in obligations_data]
    topic_counts = {t: topics.count(t) for t in set(topics)}
    print("\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")
    
    risk_levels = [o["risk_level"] for o in obligations_data]
    risk_counts = {r: risk_levels.count(r) for r in set(risk_levels)}
    print("\nRisk distribution:")
    for risk, count in sorted(risk_counts.items()):
        print(f"  {risk}: {count}")

if __name__ == "__main__":
    main()