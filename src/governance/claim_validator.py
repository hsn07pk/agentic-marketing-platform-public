import logging
import re
import yaml
import csv
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)

class ClaimValidator:

    def __init__(self):
        self.claims_library = self._load_claims_library()

    def _load_claims_library(self) -> Dict[str, Dict[str, Any]]:
        """
        Load claims library from CSV (preferred) or YAML (fallback)

        Priority:
        1. data/claim_library/claims.csv
        2. config/prompts/claim_library.yaml
        """
        csv_file = Path("data/claim_library/claims.csv")
        if csv_file.exists():
            try:
                claims_dict = {}
                with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row.get('id'):
                            continue

                        claim_id = row['id']

                        personas_str = row.get('personas', '[]')
                        tags_str = row.get('tags', '[]')

                        # Handle list format like "[decision_maker, practitioner]"
                        if personas_str.startswith('[') and not personas_str.startswith('["'):
                            personas_str = personas_str.replace('[', '["').replace(']', '"]').replace(', ', '", "')
                        if tags_str.startswith('[') and not tags_str.startswith('["'):
                            tags_str = tags_str.replace('[', '["').replace(']', '"]').replace(', ', '", "')

                        try:
                            personas = json.loads(personas_str)
                            tags = json.loads(tags_str)
                        except json.JSONDecodeError:
                            personas = []
                            tags = []

                        claims_dict[claim_id] = {
                            'id': claim_id,
                            'text': row.get('claim_text', ''),
                            'type': row.get('claim_type', ''),
                            'personas': personas,
                            'tags': tags,
                            'source': row.get('source_title', ''),
                            'source_url': row.get('source_url', ''),
                            'source_date': row.get('source_date', ''),
                            'evidence_url': row.get('source_url', ''),
                            'evidence_excerpt': row.get('evidence_excerpt', ''),
                            'confidence': int(row.get('confidence', 3))
                        }

                logger.info(f"✅ Loaded {len(claims_dict)} claims from CSV: {csv_file}")
                return claims_dict

            except Exception as e:
                logger.error(f"Failed to load claims from CSV, falling back to YAML: {e}")

        try:
            claims_file = Path("config/prompts/claim_library.yaml")

            with open(claims_file, 'r') as f:
                data = yaml.safe_load(f)

            claims_dict = {}
            for claim in data.get('claims', []):
                claims_dict[claim['id']] = claim

            logger.info(f"✅ Loaded {len(claims_dict)} claims from YAML: {claims_file}")
            return claims_dict

        except Exception as e:
            logger.error(f"❌ Failed to load claims library: {e}")
            return {}
    
    def validate_content(
        self,
        content_text: str,
        claims_used: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        result = {
            "is_valid": False,
            "all_claims_cited": False,
            "claims_found": [],
            "citations_found": [],
            "missing_citations": [],
            "hallucinated_claims": [],
            "score": 0.0
        }

        claim_ids = self._extract_claim_ids(content_text)
        result['claims_found'] = claim_ids

        if claims_used:
            for claim_id in claims_used:
                if claim_id not in claim_ids:
                    result['missing_citations'].append(claim_id)

        for claim_id in claim_ids:
            if claim_id not in self.claims_library:
                result['hallucinated_claims'].append(claim_id)
                continue

            claim_data = self.claims_library[claim_id]

            has_citation = self._check_citation(content_text, claim_data)

            if has_citation:
                result['citations_found'].append(claim_id)
            else:
                if claim_id not in result['missing_citations']:
                    result['missing_citations'].append(claim_id)

        total_claims = len(claims_used) if claims_used else len(claim_ids)
        valid_claims = len(result['citations_found'])

        result['all_claims_cited'] = (
            len(result['missing_citations']) == 0 and
            len(result['hallucinated_claims']) == 0
        )

        if total_claims > 0:
            result['score'] = valid_claims / total_claims
            result['is_valid'] = result['score'] >= 0.8 and result['all_claims_cited']
        else:
            result['is_valid'] = False
            result['score'] = 0.0

        return result
    
    def _extract_claim_ids(self, content: str) -> List[str]:
        pattern = r'(?:CLM|CLAIM)_\d{3}'
        matches = re.findall(pattern, content)
        return list(set(matches))
    
    def _check_citation(
        self,
        content: str,
        claim_data: Dict[str, Any]
    ) -> bool:
        citation_patterns = [
            r'\[Source:',
            r'\[.*?\]',
            claim_data.get('source', ''),
            r'\(Source:',
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def get_claim_by_id(self, claim_id: str) -> Optional[Dict[str, Any]]:
        return self.claims_library.get(claim_id)
    
    def get_claims_for_persona(
        self,
        persona: str
    ) -> List[Dict[str, Any]]:
        relevant_claims = []
        
        for claim_id, claim_data in self.claims_library.items():
            if persona in claim_data.get('personas', []):
                relevant_claims.append({
                    "id": claim_id,
                    "text": claim_data.get('text'),
                    "source": claim_data.get('source'),
                    "priority": claim_data.get('priority', 5)
                })
        
        relevant_claims.sort(key=lambda x: x['priority'], reverse=True)
        
        return relevant_claims
    
    def get_claims_for_goal(
        self,
        goal: str
    ) -> List[Dict[str, Any]]:
        relevant_claims = []
        
        for claim_id, claim_data in self.claims_library.items():
            if goal in claim_data.get('goals', []):
                relevant_claims.append({
                    "id": claim_id,
                    "text": claim_data.get('text'),
                    "source": claim_data.get('source'),
                    "priority": claim_data.get('priority', 5)
                })
        
        relevant_claims.sort(key=lambda x: x['priority'], reverse=True)
        
        return relevant_claims
    
    def format_claim_for_prompt(
        self,
        claim_id: str
    ) -> str:
        claim = self.get_claim_by_id(claim_id)
        
        if not claim:
            return ""
        
        return f"{claim_id}: {claim['text']} [Source: {claim['source']}]"
    
    def validate_claim_library(self) -> Dict[str, Any]:
        issues = []
        
        required_fields = ['id', 'text', 'source', 'personas', 'goals', 'priority']
        
        for claim_id, claim_data in self.claims_library.items():
            for field in required_fields:
                if field not in claim_data:
                    issues.append(f"{claim_id} missing field: {field}")
            
            if 'evidence_url' not in claim_data:
                issues.append(f"{claim_id} missing evidence_url")
        
        return {
            "valid": len(issues) == 0,
            "total_claims": len(self.claims_library),
            "issues": issues
        }