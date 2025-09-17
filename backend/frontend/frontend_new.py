import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import time
import base64

# Configure page
st.set_page_config(
    page_title="AgenticSpark - Compliance Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://localhost:8000"

def analyze_document(file) -> Dict[str, Any]:
    """Send document to backend for analysis"""
    # Encode file data as base64
    file_data = base64.b64encode(file.getvalue()).decode('utf-8')

    # Send as form data instead of multipart
    data = {
        "file_data": file_data,
        "filename": file.name
    }

    try:
        response = requests.post(f"{BACKEND_URL}/analyze", data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None

def display_results(results: Dict[str, Any]):
    """Display analysis results in organized sections"""

    if not results:
        return

    # Executive Summary
    if "advisory" in results and results["advisory"]:
        st.header("📊 Executive Summary")
        advisory = results["advisory"]

        if "executive_summary" in advisory:
            st.info(advisory["executive_summary"])

        if "recommendations" in advisory and advisory["recommendations"]:
            st.subheader("💡 Key Recommendations")
            for i, rec in enumerate(advisory["recommendations"], 1):
                st.write(f"{i}. {rec}")

    # Overall Score
    if "overall_score" in results:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            score = results["overall_score"]
            if score >= 80:
                st.success(f"🎉 Overall Compliance Score: {score:.1f}%")
            elif score >= 60:
                st.warning(f"⚠️ Overall Compliance Score: {score:.1f}%")
            else:
                st.error(f"🚨 Overall Compliance Score: {score:.1f}%")

    # Risk Analysis
    if "risks" in results and results["risks"]:
        st.header("⚠️ Risk Assessment")

        risks_df = pd.DataFrame(results["risks"])
        if not risks_df.empty:
            # Risk distribution chart
            risk_counts = risks_df["risk"].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk details table
            st.subheader("Detailed Risk Analysis")
            display_df = risks_df[["clause_id", "risk", "framework", "status", "reason"]]
            display_df.columns = ["Clause ID", "Risk Level", "Framework", "Status", "Reason"]
            st.dataframe(display_df, use_container_width=True)

    # Classification Results
    if "classified" in results and results["classified"]:
        st.header("🏷️ Clause Classification")

        classified_df = pd.DataFrame(results["classified"])
        if not classified_df.empty:
            # Classification distribution
            type_counts = classified_df["cls_type"].value_counts()
            fig = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title="Clause Types Distribution",
                labels={"x": "Clause Type", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Classification details
            st.subheader("Classified Clauses")
            display_df = classified_df[["clause_id", "text", "cls_type", "confidence"]]
            display_df.columns = ["ID", "Text", "Type", "Confidence"]
            display_df["Confidence"] = display_df["Confidence"].round(3)
            st.dataframe(display_df, use_container_width=True)

    # Extracted Clauses
    if "clauses" in results and results["clauses"]:
        st.header("📄 Extracted Clauses")

        clauses_df = pd.DataFrame(results["clauses"])
        if not clauses_df.empty:
            st.subheader(f"Total Clauses Extracted: {len(clauses_df)}")

            # Show clauses in an expandable format
            for idx, clause in clauses_df.iterrows():
                with st.expander(f"Clause {clause.get('clause_id', idx+1)}"):
                    st.write(f"**Text:** {clause.get('text', 'N/A')}")
                    if clause.get('page'):
                        st.write(f"**Page:** {clause['page']}")
                    if clause.get('section'):
                        st.write(f"**Section:** {clause['section']}")

def main():
    st.title("🤖 AgenticSpark - AI Compliance Analyzer")
    st.markdown("---")

    st.sidebar.header("📋 About")
    st.sidebar.info("""
    **AgenticSpark** is an AI-powered compliance analysis platform that:

    🔍 **Extracts** clauses from legal documents
    🏷️ **Classifies** clauses by type and purpose
    ⚠️ **Assesses** compliance risks
    💡 **Provides** executive recommendations

    Upload a PDF document to get started!
    """)

    # File upload section
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a compliance document for analysis"
    )

    if uploaded_file is not None:
        st.success(f"📄 File uploaded: {uploaded_file.name}")

        # Analysis button
        if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing document... This may take a few moments."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)

                results = analyze_document(uploaded_file)

                if results:
                    st.success("✅ Analysis complete!")
                    display_results(results)
                else:
                    st.error("❌ Analysis failed. Please check the backend connection.")

    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Upload** a PDF compliance document
        2. **Click** "Analyze Document" to start processing
        3. **Review** the results in organized sections:
           - Executive summary and recommendations
           - Risk assessment with visualizations
           - Clause classifications
           - Extracted clauses with details
        4. **Export** results if needed for reporting

        **Note:** Make sure the backend server is running on `http://localhost:8000`
        """)

if __name__ == "__main__":
    main()