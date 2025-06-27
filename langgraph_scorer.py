import os
import json
import time
from typing import List, Dict, Tuple, TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = "gpt-4.1-mini"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

# Pydantic models for structured outputs
class TagResult(BaseModel):
    tag: str = Field(description="The tag name")
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    rationale: str = Field(description="Explanation for the tag assignment")

class ClassificationResult(BaseModel):
    label: str = Field(description="The classification label")
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    rationale: str = Field(description="Explanation for the classification")

class ColumnAnalysisResult(BaseModel):
    classification: ClassificationResult
    tags: List[TagResult]

# State management for LangGraph
class ColumnState(TypedDict):
    column_name: str
    column_description: str
    analysis_result: Dict
    processing_steps: List[str]
    processing_time: float
    processing_mode: str
    error_message: str

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
    processing_mode: str = "langgraph"

class LangGraphColumnScorer:
    def __init__(self, api_key: str = None):
        # Initialize Azure OpenAI LLM for LangChain
        self.llm = AzureChatOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            azure_deployment=deployment,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Controlled vocabulary - same as original
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
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for column analysis"""
        workflow = StateGraph(ColumnState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("classify_column", self._classify_column)
        workflow.add_node("tag_column", self._tag_column)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("finalize", self._finalize_results)
        
        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "classify_column")
        workflow.add_edge("classify_column", "tag_column")
        workflow.add_edge("tag_column", "validate_results")
        workflow.add_edge("validate_results", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initialize_analysis(self, state: ColumnState) -> ColumnState:
        """Initialize the analysis process"""
        state["processing_steps"] = ["Analysis initialized"]
        state["analysis_result"] = {}
        state["error_message"] = ""
        return state
    
    def _classify_column(self, state: ColumnState) -> ColumnState:
        """Classify the column using LLM"""
        try:
            labels_str = ", ".join(self.classification_labels)
            
            classification_prompt = ChatPromptTemplate.from_template("""
You are a data classification specialist. Analyze the column and provide a classification.

Column Name: {column_name}
Column Description: {column_description}

Available Classifications: {classifications}

Provide your response in JSON format:
{{
  "classification": {{
    "label": "<classification>",
    "score": 0.0-1.0,
    "rationale": "<explanation>"
  }}
}}

Return ONLY valid JSON, no additional text.
            """)
            
            chain = classification_prompt | self.llm | JsonOutputParser()
            
            result = chain.invoke({
                "column_name": state["column_name"],
                "column_description": state["column_description"],
                "classifications": labels_str
            })
            
            state["analysis_result"]["classification"] = result.get("classification", {})
            state["processing_steps"].append("Classification completed")
            
        except Exception as e:
            state["error_message"] = f"Classification error: {str(e)}"
            state["analysis_result"]["classification"] = {
                "label": "Uncategorized",
                "score": 0.1,
                "rationale": f"Error during classification: {str(e)}"
            }
        
        return state
    
    def _tag_column(self, state: ColumnState) -> ColumnState:
        """Tag the column using LLM"""
        try:
            # Create categorized tags string
            categorized_tags_str = ""
            for category, tags in self.tag_groups.items():
                categorized_tags_str += f"\n  {category}: {', '.join(tags)}"
            
            tagging_prompt = ChatPromptTemplate.from_template("""
You are a data tagging specialist. Analyze the column and suggest relevant tags.

Column Name: {column_name}
Column Description: {column_description}

Available Tags by Category:{tags}

Instructions:
- Select 3-5 most relevant tags from ANY category
- Provide confidence scores based on how certain you are
- Use "uncategorized" only when no other tags fit

Provide your response in JSON format:
{{
  "tags": [
    {{
      "tag": "<tag_name>",
      "score": 0.0-1.0,
      "rationale": "<explanation>"
    }}
  ]
}}

Return ONLY valid JSON, no additional text.
            """)
            
            chain = tagging_prompt | self.llm | JsonOutputParser()
            
            result = chain.invoke({
                "column_name": state["column_name"],
                "column_description": state["column_description"],
                "tags": categorized_tags_str
            })
            
            state["analysis_result"]["tags"] = result.get("tags", [])
            state["processing_steps"].append("Tagging completed")
            
        except Exception as e:
            state["error_message"] = f"Tagging error: {str(e)}"
            state["analysis_result"]["tags"] = [{
                "tag": "uncategorized",
                "score": 0.1,
                "rationale": f"Error during tagging: {str(e)}"
            }]
        
        return state
    
    def _validate_results(self, state: ColumnState) -> ColumnState:
        """Validate and clean up the analysis results"""
        try:
            # Ensure classification exists
            if "classification" not in state["analysis_result"]:
                state["analysis_result"]["classification"] = {
                    "label": "Uncategorized",
                    "score": 0.1,
                    "rationale": "No classification provided"
                }
            
            # Ensure tags exist
            if "tags" not in state["analysis_result"] or not state["analysis_result"]["tags"]:
                state["analysis_result"]["tags"] = [{
                    "tag": "uncategorized",
                    "score": 0.1,
                    "rationale": "No tags provided"
                }]
            
            # Limit to top 5 tags
            state["analysis_result"]["tags"] = state["analysis_result"]["tags"][:5]
            
            state["processing_steps"].append("Results validated")
            
        except Exception as e:
            state["error_message"] = f"Validation error: {str(e)}"
        
        return state
    
    def _finalize_results(self, state: ColumnState) -> ColumnState:
        """Finalize the analysis results"""
        state["processing_steps"].append("Analysis finalized")
        return state
    
    def score_column(self, column_name: str, description: str) -> ColumnScoringResult:
        """Score a single column using LangGraph workflow"""
        start_time = time.time()
        
        # Initialize state
        initial_state = ColumnState(
            column_name=column_name,
            column_description=description,
            analysis_result={},
            processing_steps=[],
            processing_time=0.0,
            processing_mode="langgraph",
            error_message=""
        )
        
        # Execute workflow
        final_state = self.workflow.invoke(initial_state)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Convert results to original format
        try:
            classification_data = final_state["analysis_result"].get("classification", {})
            classification = ScoredResult(
                label=classification_data.get("label", "Uncategorized"),
                score=float(classification_data.get("score", 0.0)),
                rationale=classification_data.get("rationale", "No rationale provided")
            )
            
            tags_data = final_state["analysis_result"].get("tags", [])
            top_tags = []
            for tag_data in tags_data:
                tag_result = ScoredResult(
                    label=tag_data.get("tag", "uncategorized"),
                    score=float(tag_data.get("score", 0.0)),
                    rationale=tag_data.get("rationale", "No rationale provided")
                )
                top_tags.append(tag_result)
            
        except Exception as e:
            classification = ScoredResult("Uncategorized", 0.1, f"Error processing results: {str(e)}")
            top_tags = [ScoredResult("uncategorized", 0.1, "Error in result processing")]
        
        return ColumnScoringResult(
            classification=classification,
            top_tags=top_tags,
            input_prompt=f"LangGraph workflow for {column_name}",
            llm_output=json.dumps(final_state["analysis_result"], indent=2),
            processing_time=processing_time,
            processing_mode="langgraph"
        )
    
    def score_multiple_columns(self, columns: List[Tuple[str, str]], use_batch: bool = False) -> List[ColumnScoringResult]:
        """Score multiple columns using LangGraph workflow"""
        results = []
        
        for column_name, description in columns:
            result = self.score_column(column_name, description)
            results.append(result)
        
        return results
    
    def parse_json_input(self, json_input: str) -> List[Tuple[str, str]]:
        """Parse JSON input to extract column name and description pairs - same as original"""
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
    
    def get_tag_category(self, tag: str) -> str:
        """Get the category for a given tag"""
        tag_lower = tag.lower()
        for category, tags in self.tag_groups.items():
            if tag_lower in tags:
                return category
        return "Other"