import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

def create_entity_chart(entities: List[Dict[str, Any]], max_entities: int = 10) -> go.Figure:
    """
    Create a bar chart for entity prominence.
    
    Args:
        entities: List of entity dictionaries with name, type, and prominence
        max_entities: Maximum number of entities to display
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    if not entities:
        return go.Figure()
    
    # Limit to top N entities by prominence
    sorted_entities = sorted(entities, key=lambda x: x.get('prominence', 0), reverse=True)
    top_entities = sorted_entities[:max_entities]
    
    # Create DataFrame for plotting
    df = pd.DataFrame([
        {
            "Entity": e.get("name", "Unknown"),
            "Type": e.get("type", "Unknown"),
            "Prominence": e.get("prominence", 0)
        } for e in top_entities
    ])
    
    # Create chart
    fig = px.bar(
        df,
        x="Entity", 
        y="Prominence",
        color="Type",
        title=f"Top {len(top_entities)} Entities by Prominence",
        labels={"Entity": "Entity Name", "Prominence": "Prominence Score (0-1)"},
        height=500,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Entity",
        yaxis_title="Prominence Score",
        yaxis_range=[0, 1],
        legend_title="Entity Type",
        xaxis_tickangle=-45,
        margin=dict(t=50, b=100)
    )
    
    return fig

def create_topic_chart(topics: List[Dict[str, Any]], max_topics: int = 8) -> go.Figure:
    """
    Create a pie or sunburst chart for topics.
    
    Args:
        topics: List of topic dictionaries with name, category_path, confidence
        max_topics: Maximum number of topics to display
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    if not topics:
        return go.Figure()
    
    # Sort by confidence
    sorted_topics = sorted(topics, key=lambda x: x.get('confidence', 0), reverse=True)
    top_topics = sorted_topics[:max_topics]
    
    # Create DataFrame for plotting
    df = pd.DataFrame([
        {
            "Topic": t.get("name", "Unknown"),
            "Category": t.get("category_path", "/Unknown").split("/")[-1],
            "Confidence": t.get("confidence", 0),
            "Primary": "Primary" if t.get("is_primary", False) else "Secondary"
        } for t in top_topics
    ])
    
    # Create sunburst chart
    fig = px.sunburst(
        df,
        path=["Primary", "Category", "Topic"],
        values="Confidence",
        title="Topic Hierarchy",
        color="Confidence",
        color_continuous_scale=px.colors.sequential.Viridis,
        height=600
    )
    
    # Customize layout
    fig.update_layout(
        margin=dict(t=50, b=50, l=25, r=25)
    )
    
    return fig

def create_keyword_similarity_chart(similarity_results: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization for keyword similarity scores.
    
    Args:
        similarity_results: Dictionary containing scores list and average_score
        
    Returns:
        Plotly figure object
    """
    # Extract data
    scores = similarity_results.get("scores", [])
    avg_score = similarity_results.get("average_score", 0)
    
    if not scores:
        return go.Figure()
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "Keyword": s.get("keyword", "Unknown"),
            "Score": s.get("score", 0)
        } for s in scores
    ])
    
    # Sort by score
    df = df.sort_values("Score", ascending=False)
    
    # Create gauge chart for average score
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score,
        title={"text": "Average Semantic Relevance"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "rgba(50, 120, 220, 0.9)"},
            "steps": [
                {"range": [0, 30], "color": "rgba(255, 99, 132, 0.3)"},
                {"range": [30, 70], "color": "rgba(255, 205, 86, 0.3)"},
                {"range": [70, 100], "color": "rgba(75, 192, 192, 0.3)"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 70
            }
        }
    ))
    
    gauge.update_layout(
        height=300,
        margin=dict(t=30, b=0, l=30, r=30)
    )
    
    # Create bar chart for individual scores
    bar = px.bar(
        df,
        x="Keyword",
        y="Score",
        color="Score",
        color_continuous_scale=[
            (0, "rgba(255, 99, 132, 0.7)"),
            (0.3, "rgba(255, 205, 86, 0.7)"),
            (0.7, "rgba(75, 192, 192, 0.7)"),
            (1, "rgba(54, 162, 235, 0.7)")
        ],
        height=400,
        title="Semantic Relevance by Keyword"
    )
    
    bar.update_layout(
        xaxis_title="Keyword",
        yaxis_title="Relevance Score (%)",
        yaxis_range=[0, 100],
        xaxis_tickangle=-45,
        margin=dict(t=50, b=100)
    )
    
    # Return both charts
    return gauge, bar

def create_recommendations_chart(recommendations: List[Dict[str, Any]]) -> go.Figure:
    """
    Create visualization for improvement recommendations.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Plotly figure object
    """
    if not recommendations:
        return go.Figure()
    
    # Count recommendations by category and priority
    category_counts = {}
    priority_counts = {"High": 0, "Medium": 0, "Low": 0}
    
    for rec in recommendations:
        category = rec.get("category", "Other")
        priority = rec.get("priority", "Low")
        
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
        priority_counts[priority] += 1
    
    # Create category distribution chart
    category_df = pd.DataFrame([
        {"Category": cat, "Count": count} 
        for cat, count in category_counts.items()
    ])
    
    category_chart = px.bar(
        category_df,
        x="Category",
        y="Count",
        color="Category",
        title="Recommendations by Category",
        height=400
    )
    
    category_chart.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of Recommendations",
        xaxis_tickangle=-45,
        margin=dict(t=50, b=100)
    )
    
    # Create priority distribution chart
    priority_df = pd.DataFrame([
        {"Priority": pri, "Count": count} 
        for pri, count in priority_counts.items()
        if count > 0  # Only include non-zero priorities
    ])
    
    # Define custom colors for priorities
    colors = {
        "High": "rgba(255, 99, 132, 0.7)",
        "Medium": "rgba(255, 205, 86, 0.7)",
        "Low": "rgba(75, 192, 192, 0.7)"
    }
    
    priority_chart = px.pie(
        priority_df,
        values="Count",
        names="Priority",
        title="Recommendations by Priority",
        color="Priority",
        color_discrete_map=colors,
        height=400
    )
    
    priority_chart.update_layout(
        margin=dict(t=50, b=50, l=25, r=25)
    )
    
    return category_chart, priority_chart

def create_eeat_radar_chart(interpretation: Dict[str, Any]) -> go.Figure:
    """
    Create a radar chart for E-E-A-T assessment.
    
    Args:
        interpretation: Dictionary with google_interpretation data
        
    Returns:
        Plotly figure object
    """
    # Extract E-E-A-T data
    eeat = interpretation.get("eeat_assessment", {})
    
    # Define metrics and default values
    metrics = ["Expertise", "Experience", "Authoritativeness", "Trustworthiness"]
    
    # Convert text assessments to numeric values (0-10 scale)
    def text_to_score(text):
        if not text or "unable to assess" in text.lower():
            return 5  # Neutral score for unknown
        
        # Define keyword mapping to scores
        positive = ["excellent", "strong", "high", "good", "very good", "substantial", "significant"]
        negative = ["lacking", "weak", "low", "poor", "limited", "insufficient", "inadequate"]
        neutral = ["average", "moderate", "adequate", "some", "partial"]
        
        if any(word in text.lower() for word in positive):
            # Extract intensity words for finer grading
            if any(word in text.lower() for word in ["excellent", "very high", "exceptional"]):
                return 9
            return 8
        elif any(word in text.lower() for word in negative):
            # Extract intensity words for finer grading
            if any(word in text.lower() for word in ["very poor", "severely lacking"]):
                return 2
            return 3
        elif any(word in text.lower() for word in neutral):
            return 5
        
        # Default case
        return 5
    
    # Get scores for each metric
    expertise_score = text_to_score(eeat.get("expertise", ""))
    experience_score = text_to_score(eeat.get("expertise", ""))  # Using expertise as proxy
    authoritativeness_score = text_to_score(eeat.get("authoritativeness", ""))
    trustworthiness_score = text_to_score(eeat.get("trustworthiness", ""))
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[expertise_score, experience_score, authoritativeness_score, trustworthiness_score, expertise_score],  # Close the loop
        theta=metrics + [metrics[0]],  # Close the loop
        fill='toself',
        name='E-E-A-T Assessment',
        line_color='rgba(54, 162, 235, 0.8)',
        fillcolor='rgba(54, 162, 235, 0.3)'
    ))
    
    # Add reference line for average score
    fig.add_trace(go.Scatterpolar(
        r=[5, 5, 5, 5, 5],  # Average score
        theta=metrics + [metrics[0]],  # Close the loop
        fill='toself',
        name='Average',
        line=dict(color='rgba(169, 169, 169, 0.5)', dash='dot'),
        fillcolor='rgba(169, 169, 169, 0.1)'
    ))
    
    # Customize layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        title="E-E-A-T Assessment Radar",
        height=500
    )
    
    return fig

def create_semantic_overview_dashboard(analysis_results: Dict[str, Any]) -> Tuple[go.Figure, ...]:
    """
    Create a comprehensive dashboard of semantic analysis results.
    
    Args:
        analysis_results: Complete analysis results dictionary
        
    Returns:
        Tuple of Plotly figure objects
    """
    # Extract data from analysis results
    entities = analysis_results.get("entities", [])
    topics = analysis_results.get("topics", [])
    interpretation = analysis_results.get("google_interpretation", {})
    
    # Create individual charts
    entity_chart = create_entity_chart(entities)
    topic_chart = create_topic_chart(topics)
    eeat_chart = create_eeat_radar_chart(interpretation)
    
    return entity_chart, topic_chart, eeat_chart
