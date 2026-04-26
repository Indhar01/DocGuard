import streamlit as st
import os
import json
import tempfile
import zipfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
import imghdr

# Attempt to import fitz (PyMuPDF) and set a flag
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT_ENABLED = True
except ImportError:
    PDF_SUPPORT_ENABLED = False

# Import your detector (make sure the ultimate-ai-detector-combined.py is in the same directory)
try:
    from agents.ultimate_ai_detector_combined import UltimateAIContentDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    st.error("⚠️ Ultimate AI Detector not found. Please ensure ultimate-ai-detector-combined.py is in the same directory.")

# Page config
st.set_page_config(
    page_title="Ultimate AI Document Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration constants
MAX_RESULTS_IN_MEMORY = 100  # Maximum number of analysis results to keep in session state
API_CALL_DELAY = 0.5  # Delay in seconds between API calls to avoid rate limiting
MAX_PDF_PAGES = 10  # Maximum number of pages to process per PDF document
MAX_FILE_SIZE_MB = 50  # Maximum file size in megabytes

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Ensure text in metric cards is visible in both light and dark themes */
    .metric-card, .threat-severe, .threat-high, .threat-elevated, .threat-moderate, .threat-low {
        color: #31333F; /* Default dark text color for light theme */
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
    }
    
    .threat-severe {
        border-left-color: #dc3545 !important;
        background: #ffeaea;
    }
    
    .threat-high {
        border-left-color: #fd7e14 !important;
        background: #fff3e0;
    }
    
    .threat-elevated {
        border-left-color: #ffc107 !important;
        background: #fffbf0;
    }
    
    .threat-moderate {
        border-left-color: #17a2b8 !important;
        background: #e8f7fa;
    }
    
    .threat-low {
        border-left-color: #28a745 !important;
        background: #eafaf1;
    }
    
    .progress-text {
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    
    .layer-result {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #007bff;
        color: #31333F; /* Ensure text is visible */
    }
    
    .indicator-item {
        background: #fff;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-left: 3px solid #dc3545;
        color: #31333F; /* Ensure text is visible */
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'detector' not in st.session_state and DETECTOR_AVAILABLE:
    with st.spinner("🚀 Initializing Ultimate AI Detector..."):
        st.session_state.detector = UltimateAIContentDetector()

def add_results_to_session(new_results):
    """Add new results to session state with automatic cleanup of old results"""
    if not new_results:
        return
    
    # Add new results to existing results
    st.session_state.analysis_results.extend(new_results)
    
    # If we exceed the limit, keep only the most recent results
    if len(st.session_state.analysis_results) > MAX_RESULTS_IN_MEMORY:
        # Keep only the most recent MAX_RESULTS_IN_MEMORY results
        st.session_state.analysis_results = st.session_state.analysis_results[-MAX_RESULTS_IN_MEMORY:]
        st.warning(f"⚠️ Result history trimmed to most recent {MAX_RESULTS_IN_MEMORY} files to conserve memory.")

def get_threat_color(threat_level):
    """Get color based on threat level"""
    if "SEVERE" in threat_level:
        return "#dc3545"
    elif "HIGH" in threat_level:
        return "#fd7e14"
    elif "ELEVATED" in threat_level:
        return "#ffc107"
    elif "MODERATE" in threat_level:
        return "#17a2b8"
    else:
        return "#28a745"

def get_threat_class(threat_level):
    """Get CSS class based on threat level"""
    if "SEVERE" in threat_level:
        return "threat-severe"
    elif "HIGH" in threat_level:
        return "threat-high"
    elif "ELEVATED" in threat_level:
        return "threat-elevated"
    elif "MODERATE" in threat_level:
        return "threat-moderate"
    else:
        return "threat-low"

def validate_file_type(file_content, expected_extension):
    """
    Validate file type using magic numbers (file signatures)
    
    Args:
        file_content: bytes content of the file
        expected_extension: expected file extension (e.g., '.jpg', '.pdf')
    
    Returns:
        tuple: (is_valid, actual_type)
    """
    # Check PDF signature
    if file_content.startswith(b'%PDF'):
        return (expected_extension.lower() == '.pdf', 'pdf')
    
    # Check image type using imghdr for images
    # Create a temporary file for imghdr to read
    import io
    img_type = imghdr.what(None, h=file_content)
    
    if img_type:
        # Map imghdr types to extensions
        type_map = {
            'jpeg': ['.jpg', '.jpeg'],
            'png': ['.png'],
            'gif': ['.gif'],
            'tiff': ['.tiff', '.tif'],
            'bmp': ['.bmp']
        }
        
        expected_lower = expected_extension.lower()
        if img_type in type_map:
            return (expected_lower in type_map[img_type], img_type)
    
    return (False, 'unknown')

def analyze_single_file(file_path):
    """Analyze a single file"""
    try:
        if not DETECTOR_AVAILABLE:
            return {
                'error': 'Detector not available',
                'filename': os.path.basename(file_path)
            }
        
        result = st.session_state.detector.comprehensive_analysis(file_path)
        result['filename'] = os.path.basename(file_path)
        return result
    except Exception as e:
        return {
            'error': str(e),
            'filename': os.path.basename(file_path)
        }

def process_uploaded_files(uploaded_files):
    """Process multiple uploaded files, including PDF conversion using PyMuPDF"""
    results = []
    tasks_to_run = []
    temp_files_to_cleanup = []
    
    # Prepare all files for analysis
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            error_msg = f"File '{uploaded_file.name}' is too large ({file_size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
            st.error(error_msg)
            results.append({'error': error_msg, 'filename': uploaded_file.name})
            continue
        
        # Validate file type using magic numbers
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer for later reading
        
        is_valid, actual_type = validate_file_type(file_content, file_extension)
        if not is_valid:
            error_msg = f"File '{uploaded_file.name}' has invalid file type. Extension says '{file_extension}' but actual type is '{actual_type}'. Possible file corruption or extension spoofing."
            st.error(error_msg)
            results.append({'error': error_msg, 'filename': uploaded_file.name})
            continue

        if file_extension == ".pdf":
            if not PDF_SUPPORT_ENABLED:
                st.error("PDF processing is disabled. Please install 'PyMuPDF' to enable this feature.")
                results.append({'error': "PDF library (PyMuPDF) not found.", 'filename': uploaded_file.name})
                continue
            try:
                # Convert PDF pages to images using PyMuPDF
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                base_name = os.path.splitext(uploaded_file.name)[0]
                total_pages = len(doc)
                
                # Limit number of pages processed
                pages_to_process = min(total_pages, MAX_PDF_PAGES)
                if total_pages > MAX_PDF_PAGES:
                    st.warning(f"⚠️ PDF '{uploaded_file.name}' has {total_pages} pages. Processing only the first {MAX_PDF_PAGES} pages to conserve resources.")
                
                for page_num in range(pages_to_process):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    page_name = f"{base_name}_page_{page_num+1}.png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        tmp_file.write(pix.tobytes("png"))  # write raw PNG bytes
                        tasks_to_run.append({'path': tmp_file.name, 'name': page_name})
                        temp_files_to_cleanup.append(tmp_file.name)

                
                doc.close()
            except Exception as e:
                error_msg = f"Failed to process PDF '{uploaded_file.name}'. The file may be corrupt. Details: {e}"
                st.error(error_msg)
                results.append({'error': error_msg, 'filename': uploaded_file.name})
        else: # Handle standard image files
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tasks_to_run.append({'path': tmp_file.name, 'name': uploaded_file.name})
                temp_files_to_cleanup.append(tmp_file.name)

    # Run analysis on the prepared image files with guaranteed cleanup
    try:
        if tasks_to_run:
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            for i, task in enumerate(tasks_to_run):
                progress = (i) / len(tasks_to_run)
                overall_progress.progress(progress)
                status_text.text(f"🔍 Analyzing: {task['name']} ({i+1}/{len(tasks_to_run)})")
                
                try:
                    result = analyze_single_file(task['path'])
                    result['filename'] = task['name'] # Ensure correct filename is used in report
                    results.append(result)
                except Exception as e:
                    # Add error result if analysis fails
                    results.append({'error': str(e), 'filename': task['name']})
                
                # Add delay between API calls to avoid rate limiting (except for last file)
                if i < len(tasks_to_run) - 1:
                    time.sleep(API_CALL_DELAY)
            
            overall_progress.progress(1.0)
            status_text.text("✅ Analysis complete!")
    finally:
        # Guaranteed cleanup of all temp files, even if exception occurs
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                # Log but don't fail on cleanup errors
                print(f"Warning: Failed to cleanup temp file {temp_file}: {e}")
    
    return results

def display_single_result(result, container):
    """Display results for a single file"""
    with container:
        filename = result.get('filename', 'Unknown')
        
        # Header
        st.markdown(f"### 📄 {filename}")
        
        if 'error' in result and result['error']:
            st.error(f"❌ Analysis failed: {result['error']}")
            return
        
        # Overall Assessment
        overall = result.get('overall_assessment', {})
        ai_probability = overall.get('ai_probability', 0.0)
        is_ai_generated = overall.get('is_likely_ai_generated', False)
        confidence_level = overall.get('confidence_level', 'Unknown')
        threat_level = overall.get('threat_level', 'Unknown')
        recommendation = overall.get('recommendation', 'Unknown')
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card {get_threat_class(threat_level)}">
                <h3>🎯 AI Probability</h3>
                <h2 style="color: {get_threat_color(threat_level)};">{ai_probability:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_icon = "🚨" if is_ai_generated else "✅"
            status_text = "AI GENERATED" if is_ai_generated else "AUTHENTIC"
            st.markdown(f"""
            <div class="metric-card {get_threat_class(threat_level)}">
                <h3>{status_icon} Status</h3>
                <h2 style="color: {get_threat_color(threat_level)};">{status_text}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card {get_threat_class(threat_level)}">
                <h3>📊 Confidence</h3>
                <h2 style="color: {get_threat_color(threat_level)};">{confidence_level}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card {get_threat_class(threat_level)}">
                <h3>⚠️ Threat Level</h3>
                <h2 style="color: {get_threat_color(threat_level)};">{threat_level.split(' -')[0]}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendation
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 1rem;">
            <h3>📋 Recommendation</h3>
            <p style="font-size: 1.1em; margin: 0;"><strong>{recommendation}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandable detailed analysis
        with st.expander("🔍 Detailed Analysis Results", expanded=False):
            
            # Primary Indicators
            primary_indicators = overall.get('primary_indicators', [])
            if primary_indicators:
                st.markdown("<h4>🚨 Critical Security Indicators</h4>", unsafe_allow_html=True)
                for i, indicator in enumerate(primary_indicators, 1):
                    st.markdown(f"""
                    <div class="indicator-item">
                        <strong>{i}.</strong> {indicator}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Layer-by-layer results
            st.markdown("<h4>📊 Layer-by-Layer Analysis</h4>", unsafe_allow_html=True)
            
            # Layer 0: VLM Analysis
            vlm_result = result.get('layer_0_vlm_analysis', {})
            if vlm_result and 'error' not in vlm_result:
                st.markdown(f"""
                <div class="layer-result">
                    <h5>🔍 Layer 0 - VLM Analysis</h5>
                    <p><strong>Document Type:</strong> {vlm_result.get('document_type', 'Unknown')}</p>
                    <p><strong>Visual AI Score:</strong> {vlm_result.get('visual_ai_score', 0.0):.1%}</p>
                    <p><strong>Visual Summary:</strong> {vlm_result.get('visual_summary', 'N/A')[:200]}...</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Layer 1: Deepfake Detection
            deepfake_result = result.get('layer_1_deepfake_detection', {})
            if deepfake_result and 'error' not in deepfake_result:
                total_issues = 0
                analysis_results = deepfake_result.get('analysis_results', {})
                for analysis in analysis_results.values():
                    if isinstance(analysis, dict) and 'issues' in analysis:
                        total_issues += len(analysis['issues'])
                
                st.markdown(f"""
                <div class="layer-result">
                    <h5>🎯 Layer 1 - Deepfake Detection</h5>
                    <p><strong>Deepfake Confidence:</strong> {deepfake_result.get('deepfake_confidence', 0.0):.1%}</p>
                    <p><strong>Likely Deepfake:</strong> {'YES' if deepfake_result.get('is_likely_deepfake') else 'NO'}</p>
                    <p><strong>Total Issues:</strong> {total_issues}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Layer 2: Synthetic Identity
            synthetic_result = result.get('layer_2_synthetic_identity', {})
            if synthetic_result and 'error' not in synthetic_result:
                st.markdown(f"""
                <div class="layer-result">
                    <h5>🆔 Layer 2 - Synthetic Identity</h5>
                    <p><strong>Synthetic Confidence:</strong> {synthetic_result.get('synthetic_confidence', 0.0):.1%}</p>
                    <p><strong>Likely Synthetic:</strong> {'YES' if synthetic_result.get('is_likely_synthetic') else 'NO'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Layer 3: Specialized Logic
            logic_results = result.get('layer_3_specialized_logic', {})
            if isinstance(logic_results, dict) and 'error' not in logic_results:
                st.markdown('<div class="layer-result">', unsafe_allow_html=True)
                st.markdown("<h5>🔬 Layer 3 - Specialized Logic</h5>", unsafe_allow_html=True)
                for name, res in logic_results.items():
                    if isinstance(res, dict) and 'ai_confidence' in res:
                        title = name.replace('_', ' ').title()
                        if res.get('issues'):
                            st.markdown(f"⚠️ **{title}**: Issues found (Confidence: {res['ai_confidence']:.0%})")
                            for issue in res['issues']:
                                st.markdown(f"   • {issue}")
                        else:
                            st.markdown(f"✅ **{title}**: No issues detected")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Layer 5: LLM Expert Reasoning
            llm_result = result.get('layer_5_llm_reasoning', {})
            if llm_result and 'error' not in llm_result:
                st.markdown(f"""
                <div class="layer-result">
                    <h5>🧠 Layer 5 - LLM Expert Reasoning</h5>
                    <p><strong>Expert AI Probability:</strong> {llm_result.get('final_ai_probability', 0.0):.1%}</p>
                    <p><strong>Expert Conclusion:</strong> {llm_result.get('expert_conclusion', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

def create_summary_dashboard(results):
    """Create a summary dashboard for multiple results"""
    if not results:
        return
    
    st.markdown("## 📊 Analysis Summary Dashboard")
    
    # Filter out error results for statistics
    valid_results = [r for r in results if 'error' not in r and 'overall_assessment' in r]
    
    if not valid_results:
        st.warning("No valid analysis results to display.")
        return
    
    # Summary statistics
    total_files = len(results)
    valid_files = len(valid_results)
    error_files = total_files - valid_files
    
    ai_generated_count = sum(1 for r in valid_results 
                           if r.get('overall_assessment', {}).get('is_likely_ai_generated', False))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📁 Total Files Analyzed", total_files)
    
    with col2:
        st.metric("✅ Successfully Analyzed", valid_files)
    
    with col3:
        st.metric("🚨 AI Generated Detected", ai_generated_count)
    
    with col4:
        ai_percentage = (ai_generated_count / valid_files * 100) if valid_files > 0 else 0
        st.metric("📊 AI Generation Rate", f"{ai_percentage:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # AI Probability Distribution
        ai_probs = [r.get('overall_assessment', {}).get('ai_probability', 0) for r in valid_results]
        
        fig = px.histogram(
            x=ai_probs,
            nbins=10,
            title="AI Probability Distribution",
            labels={'x': 'AI Probability', 'y': 'Count'},
            color_discrete_sequence=['#1e3c72']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Threat Level Distribution
        threat_levels = [r.get('overall_assessment', {}).get('threat_level', 'Unknown').split(' -')[0] 
                        for r in valid_results]
        
        threat_counts = pd.Series(threat_levels).value_counts()
        
        colors = {
            'SEVERE': '#dc3545',
            'HIGH': '#fd7e14', 
            'ELEVATED': '#ffc107',
            'MODERATE': '#17a2b8',
            'LOW': '#28a745'
        }
        
        fig = px.pie(
            values=threat_counts.values,
            names=threat_counts.index,
            title="Threat Level Distribution",
            color=threat_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results Table")
    
    table_data = []
    for result in valid_results:
        overall = result.get('overall_assessment', {})
        table_data.append({
            'Filename': result.get('filename', 'Unknown'),
            'AI Probability': f"{overall.get('ai_probability', 0):.1%}",
            'AI Generated': '🚨 YES' if overall.get('is_likely_ai_generated') else '✅ NO',
            'Confidence': overall.get('confidence_level', 'Unknown'),
            'Threat Level': overall.get('threat_level', 'Unknown').split(' -')[0],
            'Recommendation': overall.get('recommendation', 'Unknown')[:50] + '...'
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

def export_results(results):
    """Export results to various formats"""
    if not results:
        return
    
    st.markdown("### 💾 Export Results")
    
    export_format = st.selectbox(
        "Choose export format:",
        ["JSON", "CSV", "Detailed Report"]
    )
    
    if st.button("📥 Generate Export"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "JSON":
            # Export as JSON
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_data,
                file_name=f"ai_detection_results_{timestamp}.json",
                mime="application/json"
            )
        
        elif export_format == "CSV":
            # Export as CSV
            valid_results = [r for r in results if 'error' not in r and 'overall_assessment' in r]
            table_data = []
            
            for result in valid_results:
                overall = result.get('overall_assessment', {})
                table_data.append({
                    'filename': result.get('filename', 'Unknown'),
                    'ai_probability': overall.get('ai_probability', 0),
                    'is_ai_generated': overall.get('is_likely_ai_generated', False),
                    'confidence_level': overall.get('confidence_level', 'Unknown'),
                    'threat_level': overall.get('threat_level', 'Unknown'),
                    'recommendation': overall.get('recommendation', 'Unknown'),
                    'analysis_timestamp': result.get('timestamp', 'Unknown')
                })
            
            df = pd.DataFrame(table_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=f"ai_detection_results_{timestamp}.csv",
                mime="text/csv"
            )
        
        elif export_format == "Detailed Report":
            # Export detailed report
            report_lines = []
            report_lines.append("ULTIMATE AI DOCUMENT DETECTION BATCH REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Total Files Analyzed: {len(results)}")
            report_lines.append("")
            
            for i, result in enumerate(results, 1):
                if DETECTOR_AVAILABLE and 'error' not in result:
                    report = st.session_state.detector.generate_ultimate_report(result)
                    report_lines.append(f"RESULT {i} for {result.get('filename', 'N/A')}:")
                    report_lines.append(report)
                    report_lines.append("\n" + "="*60 + "\n")
                else:
                    report_lines.append(f"RESULT {i} for {result.get('filename', 'N/A')}: ERROR - {result.get('error', 'Unknown error')}")
                    report_lines.append("")
            
            report_text = "\n".join(report_lines)
            
            st.download_button(
                label="📋 Download Detailed Report",
                data=report_text,
                file_name=f"ai_detection_detailed_report_{timestamp}.txt",
                mime="text/plain"
            )

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Ultimate AI Document Detector</h1>
        <p>Advanced GenAI + Anti-FraudGPT Detection System</p>
        <p>Multi-layer analysis with 90%+ accuracy across all fraud types</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not DETECTOR_AVAILABLE:
        st.error("❌ Detector not available. Please ensure the Ultimate AI Detector module is properly installed.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Detection settings (Simplified to one mode)
        st.markdown("### 🎯 Analysis Mode")
        st.info("The system performs a **Standard Analysis**, utilizing all detection layers for comprehensive results.")
        
        # File type filter
        st.markdown("### 📁 Supported File Types")
        st.info("✅ JPEG, PNG, TIFF, BMP\n✅ PDF")
        
        if not PDF_SUPPORT_ENABLED:
            st.warning("PDF support is disabled. Install `PyMuPDF` to enable it.")

        
        # System status
        st.markdown("### 🔧 System Status")
        if hasattr(st.session_state, 'detector'):
            genai_status = "🟢 Active" if st.session_state.detector.genai_enabled else "🟡 Disabled"
            st.info(f"GenAI Features: {genai_status}")
            
            # Health check button
            if st.button("🏥 Run Health Check"):
                with st.spinner("Testing Watsonx connection..."):
                    health_status = st.session_state.detector.test_watsonx_connection()
                    
                    if health_status['configured']:
                        if health_status['credentials_valid']:
                            st.success("✅ Watsonx connection successful!")
                            
                            # Show detailed status
                            status_details = []
                            if health_status['llm_accessible']:
                                status_details.append("✅ LLM accessible")
                            else:
                                status_details.append("❌ LLM not accessible")
                            
                            if health_status['vlm_accessible']:
                                status_details.append("✅ VLM accessible")
                            else:
                                status_details.append("❌ VLM not accessible")
                            
                            st.info("\n".join(status_details))
                            
                            if health_status['config']:
                                config = health_status['config']
                                st.code(f"""Configuration:
URL: {config['url']}
Project ID: {config['project_id']}
LLM Model: {config['llm_model']}
VLM Model: {config['vlm_model']}""")
                        else:
                            st.error("❌ Connection failed!")
                    else:
                        st.error("❌ Watsonx not configured!")
                    
                    # Show errors if any
                    if health_status['errors']:
                        for error in health_status['errors']:
                            st.error(f"Error: {error}")
                    
                    # Show warnings if any
                    if health_status['warnings']:
                        for warning in health_status['warnings']:
                            st.warning(f"Warning: {warning}")
            
            st.info("🔍 Detection Layers:\n" +
                   "• VLM Document Classification\n" +
                   "• Deepfake Detection\n" +
                   "• Synthetic Identity Validation\n" +
                   "• Specialized Logic Analysis\n" +
                   "• Traditional Forensics\n" +
                   "• LLM Expert Reasoning")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "💾 Export & Reports"])
    
    with tab1:
        st.markdown("## 📤 Upload Documents for Analysis")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose document files to analyze:",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'pdf'],
            accept_multiple_files=True,
            help="You can upload multiple image and PDF files at once."
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Show uploaded files
            with st.expander("📋 Uploaded Files", expanded=True):
                for file in uploaded_files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📄 {file.name}")
                    with col2:
                        st.write(f"{file.size / 1024:.1f} KB")
                    with col3:
                        st.write(file.type)
            
            # Analysis options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Analyze All Files", type="primary", use_container_width=True):
                    with st.spinner("🔍 Processing files... This may take a moment."):
                        results = process_uploaded_files(uploaded_files)
                        # Use helper function to add results with automatic memory management
                        add_results_to_session(results)
                        st.success("✅ Analysis complete!")
                        # Use st.rerun() to immediately switch to the results tab after analysis
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All Files", use_container_width=True):
                    st.session_state.analysis_results = []
                    st.rerun()
        
        # Display individual results
        if st.session_state.analysis_results:
            st.markdown("## 📋 Individual Analysis Results")
            
            for i, result in enumerate(st.session_state.analysis_results):
                with st.container():
                    display_single_result(result, st.container())
                    if i < len(st.session_state.analysis_results) - 1:
                        st.markdown("---")
    
    with tab2:
        if st.session_state.analysis_results:
            create_summary_dashboard(st.session_state.analysis_results)
        else:
            st.info("📊 Upload and analyze files to see the dashboard.")
    
    with tab3:
        if st.session_state.analysis_results:
            export_results(st.session_state.analysis_results)
        else:
            st.info("💾 Analyze files first to enable export functionality.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔍 Ultimate AI Document Detector v1.3 | Built with Streamlit</p>
        <p>Advanced detection using GenAI + Anti-FraudGPT techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()