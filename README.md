
# DocGuard: AI-Powered Document Authenticity Verification

**An advanced security framework built on IBM watsonx.ai and deployed on IBM Cloud Code Engine to detect and prevent fraud from AI-generated documents.**

[![IBM watsonx](https://img.shields.io/badge/IBM-watsonx-blue)](https://www.ibm.com/watsonx)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b)](https://streamlit.io/)
[![IBM Cloud](https://img.shields.io/badge/IBM_Cloud-Code_Engine-blueviolet)](https://www.ibm.com/cloud/code-engine)



### Table of Contents

1.  [Project Abstract](#1-project-abstract)
2.  [The Problem: A New Generation of Digital Fraud](#2-the-problem-a-new-generation-of-digital-fraud)
3.  [Our Solution: The DocGuard Framework](#3-our-solution-the-docguard-framework)
4.  [How We Used the IBM watsonx Platform](#4-how-we-used-the-ibm-watsonx-platform)
5.  [Technical Architecture & Workflow](#5-technical-architecture--workflow)
6.  [The "DocGuard Skill Set": A Multi-Layered Defense](#6-the-docguard-skill-set-a-multi-layered-defense)
7.  [How to Run the Solution Locally](#7-how-to-run-the-solution-locally)
8.  [Deployment on IBM Cloud Code Engine](#8-deployment-on-ibm-cloud-code-engine)
9.  [Business Value and Impact](#9-business-value-and-impact)
10. [Team Information](#10-team-information)

---

### 1. Project Abstract

Enterprises face an escalating threat from AI-generated fraudulent documents, which can infiltrate critical financial workflows like SAP, leading to significant financial loss and operational disruption. Project DocGuard is an advanced security framework designed to neutralize this threat by automating document authenticity verification in real-time. At its core is the **'Ultimate AI Document Detector,'** a multi-layered engine that leverages **IBM Watsonx.ai** for sophisticated analysis and is deployed as a scalable application on **IBM Cloud Code Engine**.

The system orchestrates a suite of deep forensic 'skills'—from pixel-level deepfake detection to synthetic identity validation—and uses IBM's Generative AI models to classify documents, perform visual analysis, and provide expert reasoning on the combined evidence. This solution stops fraud at the source, ensuring process integrity and enabling enterprises to automate their financial operations with confidence.

### 2. The Problem: A New Generation of Digital Fraud

Enterprises depend on systems like SAP to manage the backbone of their financial operations. The security of these workflows hinges on the authenticity of the documents they process. The emergence of sophisticated Generative AI tools creates a critical vulnerability: the ability for anyone to generate fraudulent documents (invoices, receipts, shipping notices) that are nearly perfect replicas of real ones.

This introduces severe business risks:
*   **Direct Financial Fraud**: Processing and paying fake invoices or reimbursement claims can lead to substantial, direct financial losses.
*   **Operational Gridlock**: The threat of fraud may force a reliance on slow, costly, and fallible manual inspection for every document, creating delays in critical payment cycles.
*   **Compliance and Audit Failures**: The inability to prevent fraudulent documents from entering the system can lead to serious regulatory non-compliance and failed audits.
*   **Erosion of Process Trust**: When the authenticity of any document is in question, trust in the automated systems that drive the business is fundamentally undermined.

### 3. Our Solution: The DocGuard Framework

DocGuard is a comprehensive solution centered on intelligent, multi-layered analysis. It consists of two main components:

1.  **The Ultimate AI Document Detector**: A powerful Python backend that serves as the analysis engine. It integrates multiple detection "skills" to perform a deep forensic analysis of each document.
2.  **Streamlit Web Interface**: A user-friendly front-end that allows users to upload single or multiple documents (including PDFs), view detailed analysis results for each file, and see a summary dashboard of batch processing.

The solution moves beyond simple checks and implements a sophisticated, weighted scoring system that considers evidence from every layer of analysis to produce a final, reliable authenticity verdict.

### 4. How We Used the IBM watsonx Platform

IBM watsonx.ai is the cognitive core of DocGuard, providing the advanced reasoning and visual analysis capabilities that make our solution so effective. We utilize the `ibm-watsonx-ai` library to integrate two powerful foundation models for critical tasks:

1.  **Layer 0: VLM-based Document Analysis (`meta-llama/llama-3-2-90b-vision-instruct`)**
    *   **What it does**: This Vision-Language Model (VLM) is the first skill to analyze the document. It performs three tasks simultaneously:
        1.  **Document Classification**: Identifies the document type (e.g., `european_retail_receipt`, `indian_tax_invoice`) to trigger the correct specialized logic.
        2.  **Full OCR Extraction**: Transcribes all text from the image, which is then used by other analysis layers.
        3.  **Visual Anomaly Detection**: Provides an initial assessment of visual artifacts, such as unnatural fonts, alignment issues, or lighting inconsistencies, generating a `visual_ai_score`.

2.  **Layer 5: LLM-based Expert Reasoning (`meta-llama/llama-3-3-70b-instruct`)**
    *   **What it does**: After all the forensic data has been gathered from the other layers, this Large Language Model (LLM) acts as a senior fraud expert.
    *   **How it works**: It receives a JSON object containing all the evidence—deepfake scores, synthetic identity flags, business logic errors—and weighs it to produce a final, refined `final_ai_probability` and a human-readable `expert_conclusion` that explains the most critical evidence.

By using watsonx for these GenAI-powered skills, DocGuard can understand and reason about documents with a level of sophistication that traditional tools cannot match.

### 5. Technical Architecture & Workflow

The DocGuard workflow is a sequence of orchestrated steps designed for maximum accuracy.

1.  **Upload**: A user uploads one or more documents via the Streamlit interface.
2.  **VLM Analysis (Layer 0)**: The document is sent to the watsonx VLM for classification, OCR, and initial visual analysis.
3.  **Parallel Forensic Analysis (Layers 1-4)**: The system triggers a multi-pronged analysis in parallel:
    *   **Deepfake Detection (Layer 1)**: Analyzes pixel patterns, frequency domains, and noise to detect signs of AI generation.
    *   **Synthetic Identity (Layer 2)**: Scrutinizes extracted text for fake names, addresses, and inconsistent data patterns.
    *   **Specialized Logic (Layer 3)**: If the VLM identified a specific document type, the corresponding logic analyzer runs (e.g., checking VAT calculations on a receipt).
    *   **Traditional Forensics (Layer 4)**: Examines metadata and compression artifacts for suspicious indicators.
4.  **LLM Reasoning (Layer 5)**: All findings are compiled and sent to the watsonx LLM for a final expert review and probability score.
5.  **Final Assessment**: The backend computes a final, weighted `ai_probability` score, threat level, and recommendation.
6.  **Display Results**: The complete analysis, from the high-level verdict to the layer-by-layer breakdown, is displayed in the Streamlit UI.

### 6. The "DocGuard Skill Set": A Multi-Layered Defense

Our solution is built on a collection of modular, independent skills that each perform a specific verification task.

*   **Skill 0: VLM Document Classification (watsonx)**: Intelligently identifies document types and performs initial visual checks.
*   **Skill 1: Document Deepfake Detection**: A suite of forensic tools that look for pixel-level evidence of AI generation, including pixel distribution analysis, frequency domain checks, and texture coherence analysis.
*   **Skill 2: Synthetic Identity Validator**: Cross-validates data extracted from the document to detect AI-generated fake personas, placeholder names, and inconsistent information.
*   **Skill 3: Specialized Logic Analyzers**: A set of rule-based modules for specific document types:
    *   `EuropeanReceiptAnalyzer`: Validates VAT calculations and checks for common AI typos.
    *   `IndianInvoiceAnalyzer`: Validates GSTIN format and checks tax calculations.
    *   `MedicalReportAnalyzer`: Checks for impossible biomedical values and validates reference ranges.
*   **Skill 4: Traditional Forensic Analysis**: Examines file metadata (EXIF data) and compression levels for clues about the document's origin.
*   **Skill 5: LLM Expert Reasoner (watsonx)**: The final decision-making skill that synthesizes all evidence to provide a definitive conclusion.

### 7. How to Run the Solution Locally

**Prerequisites:**
*   Python 3.9+
*   An IBM Cloud account with a provisioned watsonx.ai instance.
*   Your IBM Cloud API Key and watsonx.ai Project ID.

**Setup Instructions:**
1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create the environment file:**
    Create a file named `.env` in the root of the project directory and add your credentials:
    ```
    WATSONx_URL=https://us-south.ml.cloud.ibm.com
    WATSONX_API_KEY="your_ibm_cloud_api_key"
    WATSONx_PROJECT_ID="your_watsonx_project_id"
    ```
    
    **Optional configuration:**
    ```
    VLM_MODEL_ID="meta-llama/llama-3-2-90b-vision-instruct"  # Default VLM model for document classification
    LLM_MODEL_ID="meta-llama/llama-3-3-70b-instruct"         # Default LLM model for expert reasoning
    ```
    
    **Note:** The legacy environment variable name `WO_API_KEY` is still supported for backward compatibility but is deprecated. Please use `WATSONX_API_KEY` for new deployments.

**Running the Application:**
1.  Open your terminal in the project root.
2.  Run the Streamlit application:
    ```bash
    streamlit run streamlit_ai_detector_app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

### 8. Deployment on IBM Cloud Code Engine

This application is designed to be deployed as a scalable, serverless application using IBM Cloud Code Engine.

**Prerequisites:**
*   IBM Cloud CLI installed and configured.
*   A Dockerfile in the project root to containerize the application.

**Deployment Steps:**
1.  **Log in to IBM Cloud:**
    ```bash
    ibmcloud login
    ```

2.  **Target a resource group and region:**
    ```bash
    ibmcloud target -g [your-resource-group] -r [your-region]
    ```

3.  **Create or select a Code Engine project:**
    ```bash
    ibmcloud ce project select -n [your-ce-project-name] --create
    ```

4.  **Deploy the application:**
    This command builds the container image from your source code and deploys it as a Code Engine application.
    ```bash
    ibmcloud ce app create --name docguard-app --build-source . --strategy Dockerfile --env-from-secret [your-credentials-secret]
    ```
    *(Note: You should first create a secret in Code Engine to securely store your `.env` variables.)*

### 9. Business Value and Impact

Implementing the DocGuard framework provides transformative benefits:

*   **Proactive Fraud Prevention**: Stops fraudulent payments before they happen by catching fake documents at the source.
*   **Maximized Efficiency**: Automates the verification process, allowing legitimate documents to flow through the system without delay and freeing up human experts to focus only on high-risk exceptions.
*   **Strengthened Governance**: Creates a fully auditable, automated record of every verification check, dramatically improving compliance and internal controls.
*   **Scalable and Adaptable**: The skill-based architecture allows new detection methods to be easily plugged into the workflow as AI fraud techniques evolve, ensuring the solution remains effective over the long term.



