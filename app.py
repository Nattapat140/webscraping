import streamlit as st
import requests
from bs4 import BeautifulSoup
from google import genai
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import json
from dotenv import load_dotenv
from utils import create_pdf, clean_html_content

# Load environment variables
# Access the secret securely
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure the Gemini library
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuration
st.set_page_config(page_title="AI Web Scraper & Visualizer", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #1e293b);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #334155;
        color: #ffffff;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 800;
    }
    
    /* Cards/Containers */
    .css-1y4p8pa {
        padding: 2rem;
        border-radius: 12px;
        background-color: #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.title("🤖 AI Web Scraper & Visualizer")
st.markdown("Extract hardware specifications and visualize data efficiently")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Settings")
    
    if not GOOGLE_API_KEY:
        st.warning("⚠️ API Key missing! Check .env file.")
        api_key_input = st.text_input("Or enter API Key here", type="password")
        if api_key_input:
            GOOGLE_API_KEY = api_key_input
    
    if GOOGLE_API_KEY:
        st.success("API Key Loaded Successfully")


# --- Main Workflow ---

# Step 1: Input URL
target_url = st.text_input("🔗 Enter Target URL", placeholder="https://example.com/products")

if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None

if st.button("🚀 Scrape & Analyze"):
    if not target_url:
        st.error("Please enter a URL.")
    elif not GOOGLE_API_KEY:
        st.error("Please configure the API Key.")
    else:
        with st.spinner("Scraping website..."):
            try:
                # 1. Scrape
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(target_url, headers=headers)
                response.raise_for_status()
                
                # 2. Clean HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                cleaned_text = clean_html_content(soup)
                
                st.session_state.scraped_data = cleaned_text
                
                # 3. Gemini Processing
                st.info("Processing data...")
                
                # Initialize Client (New SDK)
                client = genai.Client(api_key=GOOGLE_API_KEY)
                
                extraction_prompt = f"""
                **Objective:**
                Extract key product details from the data and present them in a clean, standardized format. You must prioritize factual accuracy and clarity.

                Instructions:
                Identify: Locate every distinct product model in the text.
                Extract: For each model, pull the specific hardware data points listed below.
                Format: Output the data strictly following the template provided. Do not deviate from this structure.
                Tone: Purely technical and objective. Discard marketing adjectives (e.g., "amazing," "world-class").
                Output Template: For each product found, use this exact Markdown format:

                [Insert Model Name Here]
                Overview: [Brief 1-sentence description of application/function] 
                Key Features:
                [List 1-2 distinct physical or structural features, e.g., Form Factor] 
                Technical Specs:
                Output: [Voltage & Amperage, e.g., +12V @ 83.3A]
                Power: [Total Wattage]
                Form Factor: [e.g., 1U, ATX]
                Dimension: [Length * Width * Height]
                (Repeat for next product)

                Constraints:
                If a specific spec (like Dimension) is missing in the text, omit that bullet point.
                Ensure the divider --- is used between products.
                Focus strictly on hardware specifications (Power, Dimensions, Output).
                
                Data to process:
                {cleaned_text[:30000]} 
                """

                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=extraction_prompt
                )
                st.session_state.extracted_text = response.text
                
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display Results
if st.session_state.extracted_text:
    st.subheader("📋 Product Specifications")
    
    # Scrollable container for the results
    with st.container(height=600):
        # Wrap in div with smaller font size. Newlines are crucial for markdown rendering inside HTML.
        styled_text = f"""
        <div style="font-size: 14px; line-height: 1.6;">

{st.session_state.extracted_text}

        </div>
        """
        st.markdown(styled_text, unsafe_allow_html=True)

    
    # PDF Download
    pdf_bytes = create_pdf(st.session_state.extracted_text)
    st.download_button(
        label="📥 Download Specifications PDF",
        data=pdf_bytes,
        file_name="product_specs.pdf",
        mime="application/pdf"
    )
    
    st.markdown("---")
    st.subheader("📊 Data Visualization")
    
    # 5. User Inputs for Plotting
    col1, col2 = st.columns(2)
    with col1:
        x_axis_def = st.text_input("X-Axis Data Type", value="Total Watt", placeholder="e.g. Total Watt")
        y_axis_def = st.text_input("Y-Axis Data Type", value="Power Density", placeholder="e.g. Power Density")
    
    with col2:
        notice_def = st.text_area("Notice for AI Calculation", 
                                value="Note that for the Density please use the Watt and calculate the volume. Then use those two to compute the density.",
                                height=108)

    if st.button("📉 Generate Scatter Plot"):
        if not x_axis_def or not y_axis_def:
            st.error("Please define both axes.")
        else:
            with st.spinner("Generating Chart Data..."):
                client = genai.Client(api_key=GOOGLE_API_KEY)
                
                analysis_prompt = f"""
                **Task:**
                Analyze the raw product data below regarding Delta Networking Power Supplies. 
                Extract the "Power" (in Watts) and the "Output" configuration for each model.
                Format the result as two Python lists named `y_values` and `x_values` so I can use them directly in a `matplotlib` scatter plot.
                Notice: {notice_def}

                **Requirements:**
                1. **y_values:** A list of FLOATs/INTEGERS representing {y_axis_def} (e.g., 470, 930).
                2. **x_values:** A list of FLOATs/INTEGERS representing {x_axis_def} (e.g., 10, 20, 30).
                3. **labels:** A list of strings for the Model names (e.g., "DPSN-470BP"), so I can label the points if needed.
                4. **Response Format**: STRICTLY return a valid JSON object. Do not wrap in markdown code blocks like ```json ... ```. Just the raw JSON string. Structure:
                {{
                    "x_values": [...],
                    "y_values": [...],
                    "labels": [...]
                }}

                **Input Data:**
                {st.session_state.extracted_text}
                """
                
                try:
                    response_chart = client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=analysis_prompt
                    )
                    
                    # Clean response to ensure valid JSON
                    json_str = response_chart.text.strip()
                    if json_str.startswith("```json"):
                        json_str = json_str.split("```json")[1]
                    if json_str.endswith("```"):
                        json_str = json_str.split("```")[0]
                    
                    chart_data = json.loads(json_str)
                    
                    x_vals = chart_data.get("x_values", [])
                    y_vals = chart_data.get("y_values", [])
                    lbls = chart_data.get("labels", [])
                    
                    if not x_vals or not y_vals:
                        st.error("AI could not extract valid data points.")
                    else:
                        # PLOTTING Code
                        plt.clf() 
                        items_count = len(lbls)
                        dynamic_height = max(8, items_count * 0.4) 

                        fig, ax = plt.subplots(figsize=(12, dynamic_height))
                        
                        for i in range(len(x_vals)):
                            if i < len(y_vals) and i < len(lbls):
                                color = np.random.rand(3,)
                                ax.scatter(
                                    x_vals[i], 
                                    y_vals[i], 
                                    color=color, 
                                    s=100, 
                                    alpha=0.8, 
                                    label=lbls[i]
                                )

                        ax.set_title(f'Scatter Plot Analysis: {y_axis_def} vs {x_axis_def}')
                        ax.set_xlabel(x_axis_def)
                        ax.set_ylabel(y_axis_def)
                        ax.grid(True, linestyle='--', alpha=0.5)

                        # Legend Configuration
                        ax.legend(
                            bbox_to_anchor=(1.02, 0.5), 
                            loc='center left', 
                            title="Product Models",
                            labelspacing=1
                        )

                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Download Button
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format='png', bbox_inches='tight')
                        buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Chart",
                            data=buffer,
                            file_name="scatter_plot.png",
                            mime="image/png"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating chart: {e}")


