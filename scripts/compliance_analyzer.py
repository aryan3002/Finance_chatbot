# scripts/compliance_analyzer.py
"""
Advanced compliance analysis tool that performs:
1. Risk scoring for documents
2. Compliance gap analysis
3. Regulatory mapping
4. Automated compliance checks
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from collections import defaultdict

class ComplianceAnalyzer:
    def __init__(self):
        self.regulations_map = {
            "KYC": {
                "requirements": [
                    "identity_verification",
                    "address_proof",
                    "risk_assessment",
                    "ongoing_monitoring"
                ],
                "deadlines": {
                    "initial_verification": 10,  # days
                    "high_risk_edd": 30,  # days
                    "periodic_review": 365  # days
                }
            },
            "AML": {
                "requirements": [
                    "transaction_monitoring",
                    "suspicious_activity_detection",
                    "sar_filing",
                    "training_program"
                ],
                "deadlines": {
                    "sar_filing": 30,  # days
                    "sar_filing_no_suspect": 60,  # days
                    "continuing_sar": 90  # days
                }
            },
            "GDPR": {
                "requirements": [
                    "privacy_notice",
                    "consent_management",
                    "data_portability",
                    "breach_notification"
                ],
                "deadlines": {
                    "breach_notification_authority": 3,  # days (72 hours)
                    "data_access_request": 30,  # days
                    "erasure_request": 30  # days
                }
            },
            "PCI": {
                "requirements": [
                    "encryption",
                    "access_control",
                    "vulnerability_scanning",
                    "penetration_testing"
                ],
                "deadlines": {
                    "critical_patches": 30,  # days
                    "vulnerability_scan": 90,  # days
                    "penetration_test": 365  # days
                }
            }
        }
        
        self.risk_weights = {
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1
        }
        
        self.penalty_thresholds = {
            "criminal": ["imprisonment", "criminal penalty", "felony", "prosecution"],
            "major_fine": ["million", "% of revenue", "% of turnover"],
            "minor_fine": ["thousand", "per violation", "daily fine"],
            "regulatory": ["cease and desist", "license revocation", "suspension"]
        }

    def analyze_document(self, text: str, metadata: Dict = None) -> Dict:
        """Comprehensive analysis of compliance document."""
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "compliance_topics": self.identify_topics(text),
            "requirements": self.extract_requirements(text),
            "deadlines": self.extract_deadlines(text),
            "penalties": self.extract_penalties(text),
            "risk_score": 0,
            "gaps": [],
            "recommendations": []
        }
        
        # Calculate risk score
        analysis["risk_score"] = self.calculate_risk_score(analysis)
        
        # Identify gaps
        analysis["gaps"] = self.identify_compliance_gaps(analysis)
        
        # Generate recommendations
        analysis["recommendations"] = self.generate_recommendations(analysis)
        
        return analysis
    
    def identify_topics(self, text: str) -> List[str]:
        """Identify compliance topics in text."""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            "KYC": ["know your customer", "kyc", "customer identification", "cip", "identity verification"],
            "AML": ["anti-money laundering", "aml", "money laundering", "suspicious activity"],
            "Sanctions": ["sanctions", "ofac", "embargo", "blocked persons", "sdn list"],
            "GDPR": ["gdpr", "data protection", "privacy", "personal data", "data subject rights"],
            "PCI": ["pci", "payment card", "cardholder data", "pci dss", "card security"],
            "SAR": ["suspicious activity report", "sar", "fincen", "suspicious transaction"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def extract_requirements(self, text: str) -> List[Dict]:
        """Extract specific compliance requirements."""
        requirements = []
        
        # Pattern for mandatory requirements
        requirement_patterns = [
            r"(?:must|shall|required to|obligated to)\s+([^.]+?)(?:\.|;)",
            r"(?:it is mandatory to|organizations must)\s+([^.]+?)(?:\.|;)",
            r"(?:failure to)\s+([^.]+?)(?:will|may|result)",
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                req_text = match.strip()
                if len(req_text) > 20 and len(req_text) < 500:
                    requirements.append({
                        "text": req_text,
                        "type": self.classify_requirement(req_text),
                        "mandatory": True
                    })
        
        return requirements[:20]  # Limit to top 20
    
    def classify_requirement(self, req_text: str) -> str:
        """Classify requirement type."""
        req_lower = req_text.lower()
        
        if any(word in req_lower for word in ["retain", "maintain", "keep", "store"]):
            return "retention"
        elif any(word in req_lower for word in ["report", "notify", "file", "submit"]):
            return "reporting"
        elif any(word in req_lower for word in ["verify", "identify", "authenticate"]):
            return "verification"
        elif any(word in req_lower for word in ["monitor", "review", "assess"]):
            return "monitoring"
        elif any(word in req_lower for word in ["encrypt", "secure", "protect"]):
            return "security"
        else:
            return "general"
    
    def extract_deadlines(self, text: str) -> List[Dict]:
        """Extract compliance deadlines."""
        deadlines = []
        
        # Pattern for time-based requirements
        deadline_patterns = [
            r"within\s+(\d+)\s+(days?|hours?|months?|years?)",
            r"no later than\s+(\d+)\s+(days?|hours?|months?|years?)",
            r"(\d+)\s+(days?|hours?|months?|years?)\s+(?:of|from|after)",
            r"(?:annual|monthly|quarterly|weekly|daily)\s+(?:review|report|assessment)",
        ]
        
        for pattern in deadline_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    deadlines.append({
                        "value": match[0],
                        "unit": match[1],
                        "context": text[max(0, text.find(match[0])-50):text.find(match[0])+100]
                    })
                elif isinstance(match, str):
                    deadlines.append({
                        "frequency": match,
                        "context": text[max(0, text.find(match)-50):text.find(match)+100]
                    })
        
        return deadlines[:10]  # Limit to top 10
    
    def extract_penalties(self, text: str) -> List[Dict]:
        """Extract penalty information."""
        penalties = []
        
        # Pattern for penalties
        penalty_patterns = [
            r"(?:fine|penalty)\s+(?:of\s+)?(?:up\s+to\s+)?(\$[\d,]+(?:\s+million)?|\d+%)",
            r"(?:imprisonment|jail)\s+(?:of\s+)?(?:up\s+to\s+)?(\d+\s+years?)",
            r"(?:violation|breach)\s+(?:may\s+)?result\s+in\s+([^.]+)",
        ]
        
        for pattern in penalty_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                penalty_type = self.classify_penalty(match)
                penalties.append({
                    "description": match.strip() if isinstance(match, str) else match[0],
                    "type": penalty_type,
                    "severity": self.assess_penalty_severity(match)
                })
        
        return penalties
    
    def classify_penalty(self, penalty_text: str) -> str:
        """Classify penalty type."""
        if isinstance(penalty_text, tuple):
            penalty_text = ' '.join(penalty_text)
        
        penalty_lower = penalty_text.lower()
        
        for ptype, keywords in self.penalty_thresholds.items():
            if any(kw in penalty_lower for kw in keywords):
                return ptype
        
        return "other"
    
    def assess_penalty_severity(self, penalty_text: str) -> str:
        """Assess penalty severity."""
        if isinstance(penalty_text, tuple):
            penalty_text = ' '.join(penalty_text)
        
        penalty_lower = penalty_text.lower()
        
        if any(word in penalty_lower for word in ["criminal", "imprisonment", "felony"]):
            return "CRITICAL"
        elif any(word in penalty_lower for word in ["million", "revocation", "suspension"]):
            return "HIGH"
        elif any(word in penalty_lower for word in ["thousand", "violation"]):
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_risk_score(self, analysis: Dict) -> float:
        """Calculate overall risk score (0-100)."""
        score = 0
        
        # Factor in penalties
        for penalty in analysis["penalties"]:
            if penalty["severity"] == "CRITICAL":
                score += 30
            elif penalty["severity"] == "HIGH":
                score += 20
            elif penalty["severity"] == "MEDIUM":
                score += 10
            else:
                score += 5
        
        # Factor in number of requirements
        score += min(len(analysis["requirements"]) * 2, 30)
        
        # Factor in deadlines
        for deadline in analysis["deadlines"]:
            if "value" in deadline and deadline.get("unit") in ["hours", "hour"]:
                score += 10
            elif "value" in deadline and int(deadline.get("value", 100)) <= 7:
                score += 5
        
        # Factor in topics (some are higher risk)
        high_risk_topics = ["AML", "Sanctions", "SAR"]
        for topic in analysis["compliance_topics"]:
            if topic in high_risk_topics:
                score += 10
            else:
                score += 5
        
        return min(score, 100)  # Cap at 100
    
    def identify_compliance_gaps(self, analysis: Dict) -> List[Dict]:
        """Identify potential compliance gaps."""
        gaps = []
        
        # Check for missing requirements by topic
        for topic in analysis["compliance_topics"]:
            if topic in self.regulations_map:
                expected_reqs = self.regulations_map[topic]["requirements"]
                found_reqs = {r["type"] for r in analysis["requirements"]}
                
                for expected in expected_reqs:
                    if expected not in found_reqs:
                        gaps.append({
                            "type": "missing_requirement",
                            "topic": topic,
                            "requirement": expected,
                            "severity": "MEDIUM"
                        })
        
        # Check for missing deadlines
        if len(analysis["deadlines"]) == 0 and len(analysis["requirements"]) > 5:
            gaps.append({
                "type": "missing_deadlines",
                "description": "No specific deadlines found despite multiple requirements",
                "severity": "LOW"
            })
        
        return gaps
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Based on risk score
        if analysis["risk_score"] > 70:
            recommendations.append("HIGH PRIORITY: Implement comprehensive compliance program with dedicated resources")
            recommendations.append("Conduct immediate risk assessment and remediation planning")
        elif analysis["risk_score"] > 40:
            recommendations.append("Establish regular compliance monitoring and review processes")
            recommendations.append("Ensure all critical deadlines are tracked and met")
        
        # Based on gaps
        for gap in analysis["gaps"]:
            if gap["type"] == "missing_requirement":
                recommendations.append(f"Implement {gap['requirement']} controls for {gap['topic']} compliance")
        
        # Based on penalties
        critical_penalties = [p for p in analysis["penalties"] if p["severity"] == "CRITICAL"]
        if critical_penalties:
            recommendations.append("Prioritize controls to avoid criminal penalties and imprisonment risks")
        
        # Based on deadlines
        tight_deadlines = [d for d in analysis["deadlines"] if "value" in d and int(d.get("value", 100)) <= 3]
        if tight_deadlines:
            recommendations.append("Implement automated alerting for time-critical compliance deadlines")
        
        return recommendations[:10]  # Limit to top 10

    def generate_compliance_report(self, analyses: List[Dict]) -> Dict:
        """Generate comprehensive compliance report from multiple analyses."""
        
        report = {
            "summary": {
                "total_documents": len(analyses),
                "average_risk_score": sum(a["risk_score"] for a in analyses) / len(analyses) if analyses else 0,
                "total_requirements": sum(len(a["requirements"]) for a in analyses),
                "total_gaps": sum(len(a["gaps"]) for a in analyses),
                "critical_items": []
            },
            "by_topic": defaultdict(lambda: {"count": 0, "requirements": [], "deadlines": [], "penalties": []}),
            "timeline": [],
            "risk_matrix": [],
            "action_items": []
        }
        
        # Aggregate by topic
        for analysis in analyses:
            for topic in analysis["compliance_topics"]:
                report["by_topic"][topic]["count"] += 1
                report["by_topic"][topic]["requirements"].extend(analysis["requirements"])
                report["by_topic"][topic]["deadlines"].extend(analysis["deadlines"])
                report["by_topic"][topic]["penalties"].extend(analysis["penalties"])
        
        # Build timeline of deadlines
        all_deadlines = []
        for analysis in analyses:
            for deadline in analysis["deadlines"]:
                if "value" in deadline:
                    all_deadlines.append(deadline)
        
        # Sort by urgency
        report["timeline"] = sorted(all_deadlines, 
                                   key=lambda x: int(x.get("value", 999)))[:20]
        
        # Identify critical items
        for analysis in analyses:
            if analysis["risk_score"] > 80:
                report["summary"]["critical_items"].append({
                    "source": analysis.get("metadata", {}).get("source", "Unknown"),
                    "risk_score": analysis["risk_score"],
                    "reason": "High risk score"
                })
        
        # Generate action items
        seen_actions = set()
        for analysis in analyses:
            for rec in analysis["recommendations"]:
                if rec not in seen_actions:
                    report["action_items"].append({
                        "action": rec,
                        "priority": "HIGH" if analysis["risk_score"] > 70 else "MEDIUM"
                    })
                    seen_actions.add(rec)
        
        return report


def main():
    """Main execution function."""
    analyzer = ComplianceAnalyzer()
    
    # Process sample text (you can load from files)
    sample_text = """
    Financial institutions must verify customer identity within 10 business days of account opening.
    Failure to maintain proper KYC records may result in penalties up to $1 million per violation.
    Suspicious Activity Reports must be filed within 30 calendar days of detection.
    All cardholder data must be encrypted using AES-256 encryption.
    """
    
    # Analyze document
    analysis = analyzer.analyze_document(sample_text, {"source": "sample_regulation.txt"})
    
    # Print results
    print("=" * 80)
    print("COMPLIANCE ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nRisk Score: {analysis['risk_score']}/100")
    print(f"Topics Identified: {', '.join(analysis['compliance_topics'])}")
    
    print(f"\nRequirements Found: {len(analysis['requirements'])}")
    for req in analysis['requirements'][:5]:
        print(f"  - [{req['type']}] {req['text'][:100]}...")
    
    print(f"\nDeadlines Identified: {len(analysis['deadlines'])}")
    for deadline in analysis['deadlines'][:5]:
        if "value" in deadline:
            print(f"  - {deadline['value']} {deadline['unit']}")
        else:
            print(f"  - {deadline.get('frequency', 'N/A')}")
    
    print(f"\nPenalties: {len(analysis['penalties'])}")
    for penalty in analysis['penalties']:
        print(f"  - [{penalty['severity']}] {penalty['description'][:100]}")
    
    if analysis['gaps']:
        print(f"\nCompliance Gaps: {len(analysis['gaps'])}")
        for gap in analysis['gaps']:
            print(f"  - {gap.get('description', gap.get('requirement', 'N/A'))}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save analysis
    output_path = Path("data/processed/compliance_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\nFull analysis saved to: {output_path}")


if __name__ == "__main__":
    main()