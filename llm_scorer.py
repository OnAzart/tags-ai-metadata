import os
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import os
from openai import AzureOpenAI

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"


@dataclass
class ScoredResult:
    label: str
    score: float
    rationale: str = ""

@dataclass
class ColumnScoringResult:
    classification: ScoredResult
    top_tags: List[ScoredResult]
    input_prompt: str = ""
    llm_output: str = ""
    processing_time: float = 0.0
    processing_mode: str = "individual"

class LLMColumnScorer:
    def __init__(self, api_key: str = None):
        # self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        
        # Controlled vocabulary - expandable
        self.classification_labels = [
            "PII", "Financial", "Demographic", "Behavioral", "Technical", 
            "Temporal", "Geographical", "Categorical", "Numerical", "Textual"
        ]
        
        self.tag_groups = {
            "Business Domain": [
                "finance", "hr", "sales", "marketing", "product", "customer_support",
                "logistics", "legal", "engineering", "it", "supply_chain"
            ],
            "Data Sensitivity": [
                "pii", "phi", "confidential", "restricted", "public", "internal_only"
            ],
            "Data Frequency": [
                "realtime", "hourly", "daily", "weekly", "monthly", "ad_hoc"
            ],
            "Data Type": [
                "numeric", "categorical", "text", "boolean", "datetime", "geo", "json"
            ],
            "Compliance & Regulation": [
                "gdpr", "ccpa", "hipaa", "sox", "pci_dss"
            ],
            "Quality Indicators": [
                "trusted", "uncurated", "verified", "deprecated", "draft", "golden_source"
            ],
            "Usage Purpose": [
                "reporting", "analytics", "machine_learning", "api", "dashboard",
                "etl_pipeline", "external_sharing", "billing", "monitoring"
            ],
            "Lineage or Origin": [
                "manual_input", "third_party", "legacy_system", "user_generated", "streaming_source", "erp_source"
            ],
            "Ownership & Stewardship": [
                "owned_by_finance", "owned_by_data_team", "owned_by_product", "external_vendor"
            ]
        }
        
        # Flatten all tags for backward compatibility
        self.tag_vocabulary = []
        for category_tags in self.tag_groups.values():
            self.tag_vocabulary.extend(category_tags)
        self.tag_vocabulary.append("uncategorized")
    
    def get_tag_category(self, tag: str) -> str:
        """Get the category for a given tag"""
        tag_lower = tag.lower()
        for category, tags in self.tag_groups.items():
            if tag_lower in tags:
                return category
        return "Other"
    
    
    def _get_enhanced_prompt(self, column_name: str, description: str) -> str:
        labels_str = ", ".join(self.classification_labels)
        
        # Create categorized tags string for better prompt structure
        categorized_tags_str = ""
        for category, tags in self.tag_groups.items():
            categorized_tags_str += f"\n  {category}: {', '.join(tags)}"
        
        return f"""You are a data tagging and classification assistant. Based on the column metadata and context provided below, suggest the most relevant tags and classification. For each tag, provide a confidence score between 0.0 and 1.0 and a rationale for the choice. Use only tags from the available categories. If unsure, assign the fallback tag "uncategorized" with low confidence. Return output in JSON format only.

Column Name: {column_name}
Column Description: {description}

Available Classifications: {labels_str}

Available Tags by Category:{categorized_tags_str}

Instructions:
- Select 3-5 most relevant tags from ANY category
- Include the category in your rationale when helpful
- Provide confidence scores based on how certain you are
- Use "uncategorized" only when no other tags fit

Expected Output Format:
{{
  "classification": {{
    "label": "<classification>",
    "score": 0.0-1.0,
    "rationale": "<explanation>"
  }},
  "tags": [
    {{
      "tag": "<tag_name>",
      "score": 0.0-1.0,
      "rationale": "<explanation>"
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""
    
    def _get_batch_prompt(self, columns: List[Tuple[str, str]]) -> str:
        """Generate batch prompt for processing multiple columns at once"""
        labels_str = ", ".join(self.classification_labels)
        
        # Create categorized tags string for better prompt structure
        categorized_tags_str = ""
        for category, tags in self.tag_groups.items():
            categorized_tags_str += f"\n  {category}: {', '.join(tags)}"
        
        # Format columns for batch processing
        columns_str = ""
        for i, (col_name, col_desc) in enumerate(columns, 1):
            columns_str += f"\n{i}. Column Name: {col_name}\n   Description: {col_desc}"
        
        return f"""You are a data tagging and classification assistant. Based on the column metadata provided below, analyze ALL columns in one response and suggest the most relevant tags and classification for each. For each tag, provide a confidence score between 0.0 and 1.0 and a rationale for the choice. Use only tags from the available categories. If unsure, assign the fallback tag "uncategorized" with low confidence. Return output in JSON format only.

Columns to Analyze:{columns_str}

Available Classifications: {labels_str}

Available Tags by Category:{categorized_tags_str}

Instructions:
- Analyze ALL {len(columns)} columns in one response
- Select 3-5 most relevant tags from ANY category for each column
- Include the category in your rationale when helpful
- Provide confidence scores based on how certain you are
- Use "uncategorized" only when no other tags fit

Expected Output Format:
{{
  "results": [
    {{
      "column_index": 1,
      "column_name": "<column_name>",
      "classification": {{
        "label": "<classification>",
        "score": 0.0-1.0,
        "rationale": "<explanation>"
      }},
      "tags": [
        {{
          "tag": "<tag_name>",
          "score": 0.0-1.0,
          "rationale": "<explanation>"
        }}
      ]
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""
    
    def score_column(self, column_name: str, description: str) -> ColumnScoringResult:
        """Score a single column for classification and tags using enhanced prompt"""
        
        start_time = time.time()
        
        # Get enhanced prompt
        input_prompt = self._get_enhanced_prompt(column_name, description)
        
        # Make single API call with enhanced prompt
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": input_prompt
            }],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse JSON response
        llm_output = response.choices[0].message.content.strip()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        try:
            parsed_output = json.loads(llm_output)
            
            # Extract classification
            classif_data = parsed_output.get("classification", {})
            classification = ScoredResult(
                label=classif_data.get("label", "Uncategorized"),
                score=float(classif_data.get("score", 0.0)),
                rationale=classif_data.get("rationale", "No rationale provided")
            )
            
            # Extract tags
            tags_data = parsed_output.get("tags", [])
            top_tags = []
            for tag_data in tags_data[:5]:  # Limit to top 5
                tag_result = ScoredResult(
                    label=tag_data.get("tag", "Uncategorized"),
                    score=float(tag_data.get("score", 0.0)),
                    rationale=tag_data.get("rationale", "No rationale provided")
                )
                top_tags.append(tag_result)
            
            # Ensure we have at least one tag
            if not top_tags:
                top_tags = [ScoredResult("Uncategorized", 0.1, "No tags could be determined")]
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback in case of parsing error
            classification = ScoredResult("Uncategorized", 0.1, f"Error parsing response: {str(e)}")
            top_tags = [ScoredResult("Uncategorized", 0.1, "Error in tag extraction")]
        
        return ColumnScoringResult(
            classification=classification,
            top_tags=top_tags,
            input_prompt=input_prompt,
            llm_output=llm_output,
            processing_time=processing_time,
            processing_mode="individual"
        )
    
    def score_columns_batch(self, columns: List[Tuple[str, str]]) -> List[ColumnScoringResult]:
        """Score multiple columns in a single API request (up to 5 columns per batch)"""
        if len(columns) > 5:
            raise ValueError("Maximum 5 columns per batch request")
        
        if not columns:
            return []
        
        start_time = time.time()
        
        # Get batch prompt
        input_prompt = self._get_batch_prompt(columns)
        
        # Make single API call for all columns
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": input_prompt
            }],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse JSON response
        llm_output = response.choices[0].message.content.strip()
        
        end_time = time.time()
        total_processing_time = end_time - start_time
        # Divide time equally among columns for reporting
        per_column_time = total_processing_time / len(columns)
        
        results = []
        
        try:
            parsed_output = json.loads(llm_output)
            batch_results = parsed_output.get("results", [])
            
            # Process each column result
            for i, (col_name, col_desc) in enumerate(columns):
                # Find corresponding result by index or name
                column_result = None
                for result in batch_results:
                    if (result.get("column_index") == i + 1 or 
                        result.get("column_name", "").lower() == col_name.lower()):
                        column_result = result
                        break
                
                if column_result:
                    # Extract classification
                    classif_data = column_result.get("classification", {})
                    classification = ScoredResult(
                        label=classif_data.get("label", "Uncategorized"),
                        score=float(classif_data.get("score", 0.0)),
                        rationale=classif_data.get("rationale", "No rationale provided")
                    )
                    
                    # Extract tags
                    tags_data = column_result.get("tags", [])
                    top_tags = []
                    for tag_data in tags_data[:5]:  # Limit to top 5
                        tag_result = ScoredResult(
                            label=tag_data.get("tag", "uncategorized"),
                            score=float(tag_data.get("score", 0.0)),
                            rationale=tag_data.get("rationale", "No rationale provided")
                        )
                        top_tags.append(tag_result)
                    
                    # Ensure we have at least one tag
                    if not top_tags:
                        top_tags = [ScoredResult("uncategorized", 0.1, "No tags could be determined")]
                else:
                    # Fallback if column not found in batch results
                    classification = ScoredResult("Uncategorized", 0.1, "Column not found in batch response")
                    top_tags = [ScoredResult("uncategorized", 0.1, "Batch processing error")]
                
                column_scoring_result = ColumnScoringResult(
                    classification=classification,
                    top_tags=top_tags,
                    input_prompt=input_prompt,
                    llm_output=llm_output,
                    processing_time=per_column_time,
                    processing_mode="batch"
                )
                results.append(column_scoring_result)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback in case of parsing error - create results for all columns
            for col_name, col_desc in columns:
                classification = ScoredResult("Uncategorized", 0.1, f"Batch parsing error: {str(e)}")
                top_tags = [ScoredResult("uncategorized", 0.1, "Error in batch tag extraction")]
                
                column_scoring_result = ColumnScoringResult(
                    classification=classification,
                    top_tags=top_tags,
                    input_prompt=input_prompt,
                    llm_output=llm_output,
                    processing_time=per_column_time,
                    processing_mode="batch"
                )
                results.append(column_scoring_result)
        
        return results
    
    def parse_json_input(self, json_input: str) -> List[Tuple[str, str]]:
        """Parse JSON input to extract column name and description pairs"""
        try:
            data = json.loads(json_input)
            columns = []
            
            # Support different JSON formats
            if isinstance(data, list):
                # Format: [{"name": "col1", "description": "desc1"}, ...]
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("name", item.get("column_name", ""))
                        desc = item.get("description", item.get("desc", item.get("column_description", "")))
                        if name and desc:
                            columns.append((name, desc))
            elif isinstance(data, dict):
                # Format: {"columns": [{"name": "col1", "description": "desc1"}, ...]}
                if "columns" in data:
                    for item in data["columns"]:
                        if isinstance(item, dict):
                            name = item.get("name", item.get("column_name", ""))
                            desc = item.get("description", item.get("desc", item.get("column_description", "")))
                            if name and desc:
                                columns.append((name, desc))
                else:
                    # Format: {"col1": "desc1", "col2": "desc2", ...}
                    for name, desc in data.items():
                        if isinstance(desc, str) and name and desc:
                            columns.append((name, desc))
            
            return columns
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing JSON input: {str(e)}")
    
    def score_multiple_columns(self, columns: List[Tuple[str, str]], use_batch: bool = False) -> List[ColumnScoringResult]:
        """Score multiple columns with option to use batch processing or individual processing"""
        if not columns:
            return []
        
        if use_batch:
            # Use batch processing for groups of up to 5 columns
            results = []
            for i in range(0, len(columns), 5):
                batch = columns[i:i+5]
                batch_results = self.score_columns_batch(batch)
                results.extend(batch_results)
            return results
        else:
            # Use individual processing (original method)
            results = []
            for column_name, description in columns:
                result = self.score_column(column_name, description)
                results.append(result)
            return results