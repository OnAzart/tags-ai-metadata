import streamlit as st
import os
from dotenv import load_dotenv
from llm_scorer import LLMColumnScorer, ColumnScoringResult
from typing import List, Tuple

# Load environment variables from .env file
load_dotenv()

st.set_page_config(
    page_title="Column Tag Scorer",
    page_icon="🏷️",
    layout="wide"
)

def initialize_scorer():
    """Initialize the LLM scorer with API key"""
    if 'scorer' not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Please set your OpenAI API key in .env file or Streamlit secrets")
            st.stop()
        st.session_state.scorer = LLMColumnScorer(api_key)

def display_vocabulary():
    """Display the controlled vocabulary being used"""
    scorer = st.session_state.scorer
    
    st.sidebar.header("📚 Controlled Vocabulary")

    st.sidebar.subheader("Available Tags by Category:")
    for category, tags in scorer.tag_groups.items():
        with st.sidebar.expander(f"📂 {category}", expanded=False):
            for tag in tags:
                st.text(f"• {tag}")
    
    st.sidebar.subheader("Classification Labels:")
    for label in scorer.classification_labels:
        st.sidebar.text(f"• {label}")
    
    # Add uncategorized at the bottom
    st.sidebar.text("• uncategorized")

def display_results(results: List[ColumnScoringResult], columns_data: List[Tuple[str, str]]):
    """Display scoring results in a clean format with debugging info"""
    
    for i, (result, (col_name, col_desc)) in enumerate(zip(results, columns_data)):
        # Header with timing information
        time_info = f"⏱️ {result.processing_time:.2f}s ({result.processing_mode})"
        st.subheader(f"📊 Results for: {col_name} | {time_info}")
        
        # Classification section
        st.write("**🏷️ Classification:**")
        col1, col2 = st.columns([2, 3])
        with col1:
            st.success(f"{result.classification.label}")
            st.caption(f"Confidence: {result.classification.score:.2%}")
        with col2:
            if hasattr(result.classification, 'rationale') and result.classification.rationale:
                st.caption(f"💭 {result.classification.rationale}")
        
        st.write("**🏆 Suggested Tags:**")
        
        # Display tags in a clean grid without grouping by category
        if result.top_tags:
            # Create columns for all tags
            num_tags = len(result.top_tags)
            tag_cols = st.columns(min(num_tags, 3))  # Max 3 columns for better layout
            
            for idx, tag_result in enumerate(result.top_tags):
                col_idx = idx % 3  # Wrap to next row after 3 columns
                with tag_cols[col_idx]:
                    # Use different colors based on confidence score
                    if tag_result.score >= 0.8:
                        st.success(f"✅ {tag_result.label}")
                    elif tag_result.score >= 0.6:
                        st.info(f"🔵 {tag_result.label}")
                    elif tag_result.score >= 0.4:
                        st.warning(f"🟡 {tag_result.label}")
                    else:
                        st.error(f"🔴 {tag_result.label}")
                    
                    st.caption(f"Confidence: {tag_result.score:.1%}")
                    if hasattr(tag_result, 'rationale') and tag_result.rationale:
                        with st.expander(f"💭 Why {tag_result.label}?", expanded=False):
                            st.caption(tag_result.rationale)
        
        # Debug sections - expandable
        if hasattr(result, 'input_prompt') and hasattr(result, 'llm_output'):
            with st.expander("🔍 LLM Input", expanded=False):
                st.code(result.input_prompt, language="text")
            
            with st.expander("🤖 LLM Output", expanded=False):
                st.code(result.llm_output, language="json")
        
        st.divider()

def main():
    st.title("🏷️ Column Tag Scorer")
    st.markdown("*AI-powered column classification and tagging with probability scores*")
    
    initialize_scorer()
    display_vocabulary()
    
    # Main input form
    st.header("📝 Column Input")
    
    # Initialize columns list in session state
    if 'columns_count' not in st.session_state:
        st.session_state.columns_count = 1
    
    # Initialize sample data loading flag
    if 'load_sample_data' not in st.session_state:
        st.session_state.load_sample_data = False
    
    # Initialize processing mode
    if 'use_batch_processing' not in st.session_state:
        st.session_state.use_batch_processing = False
    
    # Initialize input mode
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "form"  # "form" or "json"
    
    # Sample data section (moved before form to avoid widget state issues)
    st.subheader("💡 Sample Data")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Load Sample Data"):
            st.session_state.load_sample_data = True
            st.rerun()
    with col2:
        st.caption("Try these sample columns: user_email, purchase_amount, birth_date, session_id, product_category")
    
    # Process sample data loading
    if st.session_state.load_sample_data:
        sample_columns = [
            ("user_email", "Email address of the registered user"),
            ("purchase_amount", "Total amount spent on the purchase in USD"),
            ("birth_date", "Date of birth of the customer"),
            ("session_id", "Unique identifier for user session"),
            ("product_category", "Category classification of the product")
        ]
        st.session_state.columns_count = len(sample_columns)
        # Clear existing values first to avoid conflicts
        for i in range(20):  # Clear more than needed to be safe
            if f"name_{i}" in st.session_state:
                del st.session_state[f"name_{i}"]
            if f"desc_{i}" in st.session_state:
                del st.session_state[f"desc_{i}"]
        # Set sample data values
        for i, (name, desc) in enumerate(sample_columns):
            st.session_state[f"sample_name_{i}"] = name
            st.session_state[f"sample_desc_{i}"] = desc
        st.session_state.load_sample_data = False  # Reset flag
        st.rerun()
    
    # Add/Remove column buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add Column"):
            st.session_state.columns_count += 1
            st.rerun()
    with col2:
        if st.button("➖ Remove Last Column") and st.session_state.columns_count > 1:
            st.session_state.columns_count -= 1
            st.rerun()
    
    st.markdown(f"**Current columns: {st.session_state.columns_count}**")
    
    # Processing and Input Mode Controls
    st.subheader("⚙️ Configuration")
    config_col1, config_col2, config_col3 = st.columns([1, 1, 2])
    
    with config_col1:
        input_mode = st.selectbox(
            "Input Mode:",
            ["form", "json"],
            index=0 if st.session_state.input_mode == "form" else 1,
            key="input_mode_select"
        )
        st.session_state.input_mode = input_mode
    
    with config_col2:
        use_batch = st.checkbox(
            "Batch Processing",
            value=st.session_state.use_batch_processing,
            help="Process multiple columns in single API request (up to 5 at once)"
        )
        st.session_state.use_batch_processing = use_batch
    
    with config_col3:
        if use_batch:
            st.success("🚀 Batch mode: Faster processing for multiple columns")
        else:
            st.info("🔄 Individual mode: Process each column separately")
    
    # Input section based on selected mode
    if st.session_state.input_mode == "json":
        st.subheader("📄 JSON Input")
        st.markdown("Enter column data in JSON format:")
        
        # JSON input examples
        with st.expander("📋 JSON Format Examples", expanded=False):
            st.code('''
# Format 1: Array of objects
[
  {"name": "user_email", "description": "Email address of the user"},
  {"name": "purchase_amount", "description": "Total amount spent"}
]

# Format 2: Object with columns array
{
  "columns": [
    {"name": "user_email", "description": "Email address of the user"},
    {"name": "purchase_amount", "description": "Total amount spent"}
  ]
}

# Format 3: Simple key-value pairs
{
  "user_email": "Email address of the user",
  "purchase_amount": "Total amount spent"
}
            ''', language="json")
        
        json_input = st.text_area(
            "JSON Input:",
            height=200,
            placeholder="Enter your column data in JSON format..."
        )
        
        if st.button("🚀 Analyze JSON Input", type="primary"):
            if not json_input.strip():
                st.warning("Please enter JSON input.")
            else:
                try:
                    columns_data = st.session_state.scorer.parse_json_input(json_input)
                    if not columns_data:
                        st.warning("No valid columns found in JSON input.")
                    else:
                        with st.spinner(f"Analyzing {len(columns_data)} column(s)..."):
                            try:
                                results = st.session_state.scorer.score_multiple_columns(
                                    columns_data, 
                                    use_batch=st.session_state.use_batch_processing
                                )
                                
                                mode_text = "batch" if st.session_state.use_batch_processing else "individual"
                                total_time = sum(result.processing_time for result in results)
                                avg_time = total_time / len(results) if results else 0
                                st.success(f"✅ Successfully analyzed {len(columns_data)} column(s) using {mode_text} processing!")
                                st.info(f"⏱️ Total processing time: {total_time:.2f}s | Average per column: {avg_time:.2f}s")
                                st.header("📈 Analysis Results")
                                
                                display_results(results, columns_data)
                                
                            except Exception as e:
                                st.error(f"❌ Error during analysis: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error parsing JSON: {str(e)}")
    
    else:  # Form mode
        with st.form("column_form"):
            st.markdown("Enter your columns with their descriptions:")
            
            columns_data = []
            
            # Create input fields dynamically based on columns_count
            for i in range(st.session_state.columns_count):
                st.subheader(f"Column {i+1}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Use sample data if available, otherwise empty
                    default_name = st.session_state.get(f"sample_name_{i}", "")
                    col_name = st.text_input(f"Name:", key=f"name_{i}", value=default_name)
                with col2:
                    default_desc = st.session_state.get(f"sample_desc_{i}", "")
                    col_desc = st.text_area(f"Description:", key=f"desc_{i}", value=default_desc)
                
                if col_name.strip() and col_desc.strip():
                    columns_data.append((col_name.strip(), col_desc.strip()))
            
            submitted = st.form_submit_button("🚀 Analyze Columns", type="primary")
            
            if submitted:
                if not columns_data:
                    st.warning("Please enter at least one column name and description.")
                else:
                    with st.spinner(f"Analyzing {len(columns_data)} column(s)..."):
                        try:
                            results = st.session_state.scorer.score_multiple_columns(
                                columns_data, 
                                use_batch=st.session_state.use_batch_processing
                            )
                            
                            mode_text = "batch" if st.session_state.use_batch_processing else "individual"
                            total_time = sum(result.processing_time for result in results)
                            avg_time = total_time / len(results) if results else 0
                            st.success(f"✅ Successfully analyzed {len(columns_data)} column(s) using {mode_text} processing!")
                            st.info(f"⏱️ Total processing time: {total_time:.2f}s | Average per column: {avg_time:.2f}s")
                            st.header("📈 Analysis Results")
                            
                            display_results(results, columns_data)
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()