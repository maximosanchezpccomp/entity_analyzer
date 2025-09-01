import streamlit as st
import pandas as pd
import numpy as np
from utils.url_utils import validate_url, fetch_url_content, extract_main_content
from utils.nlp_utils import OpenAINLPEngine
from modules.url_analyzer import URLSemanticAnalyzer
from modules.keyword_scorer import KeywordSemanticScorer
from modules.improvement_advisor import SemanticImprovementAdvisor
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Page configuration
st.set_page_config(
    page_title="Semantic SEO Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem 1rem;
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 700;
        color: #1E3A8A;
    }
    .st-emotion-cache-16txtl3 h2 {
        font-weight: 600;
        color: #1E3A8A;
    }
    .dashboard-header {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .priority-high {
        color: #EF4444;
        font-weight: bold;
    }
    .priority-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .priority-low {
        color: #10B981;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'url_analysis_results' not in st.session_state:
    st.session_state.url_analysis_results = None
if 'similarity_results' not in st.session_state:
    st.session_state.similarity_results = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'main_content' not in st.session_state:
    st.session_state.main_content = ""

# Sidebar for API key input and app information
with st.sidebar:
    st.title("Semantic SEO Analyzer")
    st.subheader("Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to use the semantic analysis features.")
    st.session_state.api_key = api_key
    
    st.markdown("---")
    st.markdown("""
    ### About this App
    
    This application analyzes the semantic structure of web content to help optimize it for modern search engines. It uses advanced NLP to:
    
    1. Extract key entities and topics
    2. Score relevance for target keywords
    3. Provide actionable recommendations
    
    Powered by OpenAI's GPT models.
    """)

# Main content area
st.title("Semantic SEO Analyzer")

# URL input
url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com/page")

# Keywords input
col1, col2 = st.columns([1, 1])
with col1:
    similarity_keywords = st.text_input("Keywords for Semantic Relevance (comma-separated):", 
                                       placeholder="semantic SEO, content analysis",
                                       help="Enter keywords to check content relevance")
with col2:
    improvement_keywords = st.text_input("Target Keywords for Recommendations (comma-separated):", 
                                        placeholder="semantic analysis, entity extraction",
                                        help="Enter target keywords for optimization recommendations")

# Process button
analyze_button = st.button("Analyze URL", type="primary", use_container_width=True)

# Error handling function
def show_error(message):
    st.error(f"Error: {message}")
    return None

# Main analysis logic
if analyze_button:
    # Validate inputs
    if not st.session_state.api_key:
        show_error("Please enter your OpenAI API key in the sidebar.")
    elif not validate_url(url_input):
        show_error("Please enter a valid URL.")
    else:
        try:
            # Show loading state
            with st.spinner("Analyzing URL content..."):
                # Initialize NLP engine with API key
                nlp_engine = OpenAINLPEngine(api_key=st.session_state.api_key)
                
                # Fetch URL content
                html_content = fetch_url_content(url_input)
                main_content = extract_main_content(html_content)
                st.session_state.main_content = main_content
                
                # Initialize analyzers
                url_analyzer = URLSemanticAnalyzer(nlp_engine)
                
                # Analyze URL semantics
                analysis_results = url_analyzer.analyze_url(url_input)
                st.session_state.url_analysis_results = analysis_results
                
                # Process keywords if provided
                if similarity_keywords:
                    keyword_list = [k.strip() for k in similarity_keywords.split(",") if k.strip()]
                    if keyword_list:
                        keyword_scorer = KeywordSemanticScorer(nlp_engine)
                        similarity_results = keyword_scorer.compute_similarity(main_content, keyword_list)
                        st.session_state.similarity_results = similarity_results
                
                # Generate recommendations if target keywords provided
                if improvement_keywords and st.session_state.url_analysis_results:
                    target_keyword_list = [k.strip() for k in improvement_keywords.split(",") if k.strip()]
                    if target_keyword_list:
                        improvement_advisor = SemanticImprovementAdvisor(nlp_engine)
                        recommendations = improvement_advisor.generate_recommendations(
                            main_content, st.session_state.url_analysis_results, target_keyword_list
                        )
                        st.session_state.recommendations = recommendations
                
                st.success("Analysis completed successfully!")
        except Exception as e:
            show_error(str(e))  # Esta línea falta en el código original

# Display results if available - esta sección debe estar fuera del bloque try
if st.session_state.url_analysis_results:
    st.markdown("---")
    
    # Create tabs for different sections of results
    tab1, tab2, tab3, tab4 = st.tabs(["Semantic Profile", "Keyword Relevance", "Improvement Plan", "Raw Data"])
    
    # Tab 1: Semantic Profile
    with tab1:
        st.header("Semantic Profile Analysis")
        
        # URL metadata
        metadata = st.session_state.url_analysis_results.get("metadata", {})
        st.subheader("Page Metadata")
        metadata_col1, metadata_col2 = st.columns([1, 1])
        with metadata_col1:
            st.markdown(f"**Title:** {metadata.get('title', 'N/A')}")
        with metadata_col2:
            st.markdown(f"**Description:** {metadata.get('description', 'N/A')}")
        
        # Google's interpretation
        st.subheader("Google's Probable Interpretation")
        interpretation = st.session_state.url_analysis_results.get("google_interpretation", {})
        
        interp_col1, interp_col2 = st.columns([1, 1])
        with interp_col1:
            st.markdown(f"**Main Topic:** {interpretation.get('main_topic', 'Unknown')}")
            st.markdown("**Secondary Topics:**")
            for topic in interpretation.get('secondary_topics', [])[:5]:
                st.markdown(f"- {topic}")
        
        with interp_col2:
            eeat = interpretation.get('eeat_assessment', {})
            st.markdown("**E-E-A-T Assessment:**")
            st.markdown(f"- **Expertise:** {eeat.get('expertise', 'Unable to assess')}")
            st.markdown(f"- **Authoritativeness:** {eeat.get('authoritativeness', 'Unable to assess')}")
            st.markdown(f"- **Trustworthiness:** {eeat.get('trustworthiness', 'Unable to assess')}")
            
        # Strengths and weaknesses
        st_col1, st_col2 = st.columns([1, 1])
        with st_col1:
            st.markdown("**Semantic Strengths:**")
            for strength in interpretation.get('strengths', [])[:5]:
                st.markdown(f"- {strength}")
        
        with st_col2:
            st.markdown("**Semantic Weaknesses:**")
            for weakness in interpretation.get('weaknesses', [])[:5]:
                st.markdown(f"- {weakness}")
        
        # Entity analysis
        st.subheader("Entity Analysis")
        entities = st.session_state.url_analysis_results.get("entities", [])
        
        if entities:
            # Prepare data for table
            entity_data = []
            for e in entities:
                entity_data.append({
                    "Entity": e.get("name", ""),
                    "Type": e.get("type", ""),
                    "Prominence": e.get("prominence", 0),
                    "Key Mentions": ", ".join(e.get("key_mentions", [])[:2])
                })
            
            # Convert to DataFrame for display
            entity_df = pd.DataFrame(entity_data)
            
            # Sort by prominence
            entity_df = entity_df.sort_values(by="Prominence", ascending=False).reset_index(drop=True)
            
            # Display table
            st.dataframe(entity_df, use_container_width=True)
            
            # Create entity prominence chart (for top 10 entities)
            top_entities = entity_df.head(10)
            fig = px.bar(
                top_entities, 
                x="Entity", 
                y="Prominence",
                color="Type",
                title="Top Entities by Prominence",
                labels={"Entity": "Entity Name", "Prominence": "Prominence Score (0-1)"},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entities found or extracted.")
        
        # Topic analysis
        st.subheader("Topic Analysis")
        topics = st.session_state.url_analysis_results.get("topics", [])
        
        if topics:
            # Prepare data for table
            topic_data = []
            for t in topics:
                topic_data.append({
                    "Topic": t.get("name", ""),
                    "Category Path": t.get("category_path", ""),
                    "Confidence": t.get("confidence", 0),
                    "Primary": "Yes" if t.get("is_primary", False) else "No"
                })
            
            # Convert to DataFrame for display
            topic_df = pd.DataFrame(topic_data)
            
            # Sort by confidence
            topic_df = topic_df.sort_values(by="Confidence", ascending=False).reset_index(drop=True)
            
            # Display table
            st.dataframe(topic_df, use_container_width=True)
            
            # Create topic confidence chart
            fig = px.pie(
                topic_df.head(5), 
                values="Confidence", 
                names="Topic",
                title="Top 5 Topics by Confidence",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topics found or extracted.")
        
        # Heading structure
        st.subheader("Content Structure")
        headings = st.session_state.url_analysis_results.get("headings", {})
        
        if any(headings.values()):
            for heading_level, heading_texts in headings.items():
                if heading_texts:
                    st.markdown(f"**{heading_level.upper()}:**")
                    for heading in heading_texts:
                        st.markdown(f"- {heading}")
        else:
            st.info("No heading structure found.")
    
    # Tab 2: Keyword Relevance
    with tab2:
        st.header("Semantic Keyword Relevance")
        
        if st.session_state.similarity_results:
            # Extract data
            scores = st.session_state.similarity_results.get("scores", [])
            avg_score = st.session_state.similarity_results.get("average_score", 0)
            
            # Display overall score
            st.metric("Overall Semantic Relevance", f"{avg_score:.1f}%")
            
            if scores:
                # Prepare data for table
                score_data = []
                for s in scores:
                    score_data.append({
                        "Keyword": s.get("keyword", ""),
                        "Relevance Score": s.get("score", 0),
                        "Justification": s.get("justification", "")
                    })
                
                # Convert to DataFrame
                score_df = pd.DataFrame(score_data)
                
                # Sort by score
                score_df = score_df.sort_values(by="Relevance Score", ascending=False).reset_index(drop=True)
                
                # Display table
                st.dataframe(score_df, use_container_width=True)
                
                # Create score visualization
                fig = px.bar(
                    score_df, 
                    x="Keyword", 
                    y="Relevance Score",
                    color="Relevance Score",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Semantic Relevance by Keyword",
                    labels={"Keyword": "Keyword", "Relevance Score": "Relevance Score (%)"},
                    height=400
                )
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
                
                # Keyword insights
                st.subheader("Key Insights")
                if score_df["Relevance Score"].max() > 80:
                    st.success(f"High semantic relevance for keyword: {score_df.iloc[0]['Keyword']}")
                
                if score_df["Relevance Score"].min() < 50 and len(score_df) > 1:
                    low_kw = score_df.iloc[-1]['Keyword']
                    st.warning(f"Low semantic relevance for keyword: {low_kw}")
                    
                    # Get justification if available
                    justification = score_df.iloc[-1].get("Justification", "")
                    if justification:
                        st.markdown(f"**Reason:** {justification}")
            else:
                st.info("No keyword relevance scores available.")
        else:
            st.info("Enter keywords in the 'Keywords for Semantic Relevance' field and re-analyze to see relevance scores.")
    
    # Tab 3: Improvement Plan
    with tab3:
        st.header("Semantic Improvement Plan")
        
        if st.session_state.recommendations:
            # Group recommendations by priority
            high_priority = []
            medium_priority = []
            low_priority = []
            
            for rec in st.session_state.recommendations:
                if rec["priority"] == "High":
                    high_priority.append(rec)
                elif rec["priority"] == "Medium":
                    medium_priority.append(rec)
                else:
                    low_priority.append(rec)
            
            # Display priority sections
            if high_priority:
                st.subheader("🔴 High Priority Actions")
                for i, rec in enumerate(high_priority, 1):
                    with st.expander(f"{i}. {rec['category']} - {rec['recommendation'][:50]}...", expanded=True):
                        st.markdown(f"**Recommendation:** {rec['recommendation']}")
                        st.markdown(f"**Justification:** {rec['justification']}")
                        st.markdown(f"**Category:** {rec['category']}")
            
            if medium_priority:
                st.subheader("🟠 Medium Priority Actions")
                for i, rec in enumerate(medium_priority, 1):
                    with st.expander(f"{i}. {rec['category']} - {rec['recommendation'][:50]}...", expanded=False):
                        st.markdown(f"**Recommendation:** {rec['recommendation']}")
                        st.markdown(f"**Justification:** {rec['justification']}")
                        st.markdown(f"**Category:** {rec['category']}")
            
            if low_priority:
                st.subheader("🟢 Low Priority Actions")
                for i, rec in enumerate(low_priority, 1):
                    with st.expander(f"{i}. {rec['category']} - {rec['recommendation'][:50]}...", expanded=False):
                        st.markdown(f"**Recommendation:** {rec['recommendation']}")
                        st.markdown(f"**Justification:** {rec['justification']}")
                        st.markdown(f"**Category:** {rec['category']}")
            
            # Recommendation categories visualization
            categories = [rec["category"] for rec in st.session_state.recommendations]
            category_counts = pd.Series(categories).value_counts().reset_index()
            category_counts.columns = ["Category", "Count"]
            
            fig = px.bar(
                category_counts,
                x="Category",
                y="Count",
                color="Category",
                title="Recommendations by Category",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter target keywords in the 'Target Keywords for Recommendations' field and re-analyze to see improvement recommendations.")
    
    # Tab 4: Raw Data
    with tab4:
        st.header("Raw Analysis Data")
        
        # Show the first 1000 characters of the extracted content
        st.subheader("Extracted Content Sample")
        content_preview = st.session_state.main_content[:1000] + "..." if len(st.session_state.main_content) > 1000 else st.session_state.main_content
        st.text_area("Content", content_preview, height=200)
        
        # Show raw analysis data in JSON format
        st.subheader("Raw Analysis Results")
        st.json(st.session_state.url_analysis_results)
        
        if st.session_state.similarity_results:
            st.subheader("Raw Similarity Results")
            st.json(st.session_state.similarity_results)
        
        if st.session_state.recommendations:
            st.subheader("Raw Recommendations")
            st.json(st.session_state.recommendations)

# Footer
st.markdown("---")
st.markdown("Semantic SEO Analyzer - Powered by OpenAI and Streamlit")

# Hide Streamlit branding
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
