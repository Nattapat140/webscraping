import streamlit as st
import requests
from bs4 import BeautifulSoup
from google import genai
import plotly.express as px
import pandas as pd
import re
from utils import create_pdf, create_cluster_pdf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Access the secret securely
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]

# Configuration
st.set_page_config(page_title="AI Web Scraper & Visualizer", layout="wide")

# Custom CSS
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
st.markdown("This webapp is still in testing, the cost from calling gemini is linked to Ball's billing account")

# --- Main Workflow is hereeeeeee ---

# Input URL
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
                cleaned_text = ""
                
                # 1. Scrape
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(target_url, headers=headers)
                response.raise_for_status()
                
                # 2. Clean HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                cleaned_text = soup.get_text(separator=' ', strip=True)
                
                st.session_state.scraped_data = cleaned_text
                
                # 3. Gemini Processing
                st.info("Processing data...")
                
                # Initialize Client (New SDK)
                client = genai.Client(api_key=GOOGLE_API_KEY)
                
                extraction_prompt = f"""
                **Objective:**
                Extract key product details from the data and present them in a clean, standardized format. 
                CRITICAL INSTRUCTION: You must extract EVERY SINGLE product model found in the text. Do not summarize, do not skip any, and do not use "etc." or "..." for the list of products. If there are 400 products, output 400 blocks.

                Instructions:
                Identify: Locate every distinct product model in the text.
                Extract: For each model, pull the specific hardware data points listed below.
                Format: Output the data strictly following the template provided. Do not deviate from this structure.
                Tone: Purely technical and objective. Discard marketing adjectives (e.g., "amazing," "world-class").
                Output Template: For each product found, use this exact Markdown format:

                [Insert Model Name Here]
                Key Features:
                [List 1-2 distinct physical or structural features, e.g., Form Factor] 
                Technical Specs:
                [Dynamically extract all relevant technical specifications found for this product, using the exact field names from the source text where possible]
                - [Field Name]: [Value] (Must start with a hyphen)
                - [Field Name]: [Value]
                ...
                Dimension: [Length * Width * Height] (Always prioritize extracting dimensions if available)
                CRITICAL FORMATTING FOR DIMENSION: You must strictly use the format "Num * Num * Num" (e.g., 6.95 * 5.83 * 1.55). Do not simply concatenate numbers. Ensure the asterisk * is present.
                (Repeat for next product).

                Constraints:
                If a specific spec (like Dimension) is missing in the text, omit that bullet point.
                Ensure the divider --- is used between products.
                Focus strictly on hardware technical specifications.
                
                Data to process:
                {cleaned_text[:]} 
                """

                # Stream the response
                response_stream = client.models.generate_content_stream(
                    model='gemini-2.5-flash', 
                    contents=extraction_prompt
                )
                
                # Create a generator that yields text chunks
                def stream_text():
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text

                # Display streaming output
                st.subheader("📋 Product Specifications")
                with st.container(height=600):
                    full_response = st.write_stream(stream_text())
                
                # Store final result
                st.session_state.extracted_text = full_response
                st.rerun()

                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display Results
if st.session_state.extracted_text:
    st.subheader("📋 Product Specifications")
    
    # Scrollable container for the results
    with st.container(height=600):
        # Wrap in div with smaller font size. Newlines are crucial for markdown rendering inside HTML.
        st.markdown(st.session_state.extracted_text)

    
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
    
    # User Inputs for Plotting
    # Extract keys dynamically
    extracted_keys = []
    if st.session_state.extracted_text:
        # Regex to find potential keys. 
        # Matches lines that start with optional hyphen, then some text, then a colon.
        # Captures the text before the colon.
        matches = re.findall(r"(?:^|\n)(?:\s*-\s*)?([^:\n]+):", st.session_state.extracted_text)
        
        # Blacklist of standard headers/labels to exclude
        exclude_keys = {
            "Overview", "Key Features", "Technical Specs", "Constraints", 
            "Data to process", "Output", "Input", "Note", "Warning", "Dimensions"
        }
        
        # We need to filter and clean
        unique_keys = set()
        for m in matches:
            key = m.strip()
            # exclude if it matches a known header or is too long (likely a sentence)
            if key and key not in exclude_keys and len(key) < 40 and "Overview" not in key:
                 unique_keys.add(key)
        
        extracted_keys = sorted(list(unique_keys))
    
    # Add "Others" option
    options = extracted_keys + ["Others"]

    col1, col2 = st.columns(2)
    with col1:
        # X Axis
        x_dropdown = st.selectbox("X-Axis Data Type", options=options, index=None, placeholder="Select X Axis")
        if x_dropdown == "Others":
            x_axis_def = st.text_input("Enter X-Axis manually", placeholder="e.g. Total Watt")
        else:
            x_axis_def = x_dropdown

        # Y Axis
        y_dropdown = st.selectbox("Y-Axis Data Type", options=options, index=None, placeholder="Select Y Axis")
        if y_dropdown == "Others":
            y_axis_def = st.text_input("Enter Y-Axis manually", placeholder="e.g. Power Density")
        else:
            y_axis_def = y_dropdown

    with col2:
        notice_def = st.text_area("Notice for AI Calculation", 
                                value="",
                                placeholder="e.g. Note that for the Density please use the Watt and calculate the volume...",
                                height=108)

    if st.button("📉 Generate Scatter Plot"):
        if not x_axis_def or not y_axis_def:
            st.error("Please define both axes.")
        else:
            with st.spinner("Generating Chart Data..."):
                client = genai.Client(api_key=GOOGLE_API_KEY)
                
                analysis_prompt = f"""
                **Task:**
                Analyze the raw product data below. 
                Extract the "{x_axis_def}" and "{y_axis_def}" data for each model.
                Notice: {notice_def}

                **Calculation Rule:**
                If a value for "{x_axis_def}" or "{y_axis_def}" is not directly stated, YOU MUST CALCULATE IT using available data.
                Example: If Y is "Power Density" and you have "Watt" and "Dimension", calculate Watt / Volume.

                **Response Format:**
                Output strictly as a pipe-separated list (CSV-like) with NO headers and NO markdown code blocks.
                Format:
                Model Name | X_Value | Y_Value
                Model Name | X_Value | Y_Value

                **Constraints:**
                - X_Value and Y_Value must be pure numbers (e.g. 10.5). Do not include units like 'W' or 'mm'.
                - CRITICAL: You MUST ROUND all float values to exactly 2 decimal places (e.g., 48.66, not 48.6587...).
                - Perform the calculation step-by-step internally to ensure high precision before rounding.
                - Do not use commas in numbers (e.g. use 1200 not 1,200).
                - Only output valid data lines.

                **Input Data:**
                {st.session_state.extracted_text}
                """
                try:
                    response_chart = client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=analysis_prompt
                    )
                    
                    # Parse Pipe-Separated Output
                    raw_text = response_chart.text.strip()
                    
                    x_vals = []
                    y_vals = []
                    lbls = []
                    
                    for line in raw_text.split('\n'):
                        if "|" in line:
                            parts = line.split('|')
                            if len(parts) >= 3:
                                try:
                                    model_name = parts[0].strip()
                                    # Basic cleaning just in case
                                    x_str = parts[1].strip().replace(',', '').replace('$', '')
                                    y_str = parts[2].strip().replace(',', '').replace('$', '')
                                    
                                    # Ensure they are numbers
                                    # regex to find float-like pattern if needed, but strictly float() logic is usually enough if prompt obeyed.
                                    # use a small regex to extract the first number in the string to be safe against "100 mm"
                                    x_match = re.search(r'-?\d*\.?\d+', x_str)
                                    y_match = re.search(r'-?\d*\.?\d+', y_str)
                                    
                                    if x_match and y_match:
                                        x_vals.append(float(x_match.group()))
                                        y_vals.append(float(y_match.group()))
                                        lbls.append(model_name)
                                except Exception:
                                    continue # Skip header or bad lines
                    
                    if not x_vals:
                        st.error("AI extracted no valid data points.")
                    else:                        
                        # Store in session state for persistence and toggling
                        st.session_state.chart_data = {
                            "x": x_vals,
                            "y": y_vals,
                            "labels": lbls,
                            "x_label": x_axis_def,
                            "y_label": y_axis_def
                        }
                        
                except Exception as e:
                    st.error(f"Error generating chart: {e}")

    # Render Chart if data exists (Outside the button to allow retoggling)
    if 'chart_data' in st.session_state and st.session_state.chart_data:
        data = st.session_state.chart_data
        
        # Prepare Data for Clustering
        df = pd.DataFrame({
            'Model Name': data['labels'],
            'x': data['x'],
            'y': data['y']
        })
        
        # Normalize Data (Important for K-Means)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['x', 'y']])
        
        # Perform K-Means Clustering (K=3)
        # Handle small dataset edge case
        n_clusters = min(3, len(df))
        if n_clusters < 1: n_clusters = 1
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Create Plotly Figure
        # Shift clusters to start from 1 instead of 0
        df['cluster'] = (df['cluster'] + 1).astype(str)
        
        # Explicitly map cluster IDs to Plotly Default Colors to ensure Chart and List match
        colors = px.colors.qualitative.Plotly
        # Keys are now "1", "2", "3"...
        color_map = {str(i+1): colors[i % len(colors)] for i in range(n_clusters)}
        
        # Calculate Frequency for Bubble Size (Overlap Handling)
        df['frequency'] = df.groupby(['x', 'y'])['x'].transform('count')

        # Toggle for showing labels is backkkkkk
        show_labels = st.checkbox("Show Labels", value=False)
        
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            size='frequency', # Size based on overlap count
            size_max=20, # Cap the maximum size
            hover_name='Model Name',
            text='Model Name' if show_labels else None,
            hover_data={'x': True, 'y': True, 'cluster': False, 'Model Name': False, 'frequency': True}, 
            labels={'x': data['x_label'], 'y': data['y_label'], 'cluster': 'Group', 'frequency': 'Count'},
            title=f"Clustered Analysis: {data['y_label']} vs {data['x_label']}",
            template="plotly_white", # Use white theme
            height=600,
            color_discrete_map=color_map, # Enforce colors
            category_orders={'cluster': sorted(list(color_map.keys()), key=int)} # Ensure legend order numeric
        )

        if show_labels:
            fig.update_traces(textposition='top center')

        # Update layout for premium look
        fig.update_traces(
            marker=dict(line=dict(width=1, color='DarkSlateGrey')), # Removed fixed size=12
            selector=dict(mode='markers')
        )
        fig.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(family="Inter, sans-serif", size=14, color="#000000"),
            title_font=dict(size=20, family="Inter, sans-serif", weight=800, color="#000000"),
            legend_title_text='Clusters',
            legend=dict(
                font=dict(color="#000000"),
                title=dict(font=dict(color="#000000"))
            ),
            xaxis=dict(
                title_font=dict(color="#000000"),
                tickfont=dict(color="#000000"),
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                title_font=dict(color="#000000"),
                tickfont=dict(color="#000000"),
                gridcolor='#e2e8f0'
            )
        )

        # Display with streamlit
        st.plotly_chart(fig, use_container_width=True)
            
        # Display Grouped Lists
        st.markdown("---")
        st.subheader("🧩 Product Clusters")
        
        cols = st.columns(n_clusters)
        cluster_export_data = [] # Data structure for PDF export
        
        # Group data by cluster
        for i in range(n_clusters):
            cluster_id = i+1 # 1-based index
            
            # Calculate Color
            # Use the SAME map as the chart
            hex_color = color_map[str(cluster_id)]
            
            # Convert Hex to RGB for PDF
            hex_color_stripped = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color_stripped[i:i+2], 16) for i in (0, 2, 4))
            
            # Get data for this cluster
            cluster_indices = df[df['cluster'] == str(cluster_id)].index
            
            # Prepare Lists
            models = [data['labels'][i] for i in cluster_indices]
            xs = [data['x'][i] for i in cluster_indices]
            ys = [data['y'][i] for i in cluster_indices]
            
            # Create Display DataFrame
            display_df = pd.DataFrame({
                "Model Name": models,
                data['x_label']: xs,
                data['y_label']: ys
            })

            # Collect data for PDF
            cluster_items = []
            for m, x, y in zip(models, xs, ys):
                cluster_items.append({'model': m, 'x': x, 'y': y})
            
            cluster_export_data.append({
                'id': cluster_id,
                'color': (r, g, b),
                'items': cluster_items
            })
            
            with cols[cluster_id-1]:
                # Colored Header
                st.markdown(f"<h4 style='color: {hex_color};'>Group {cluster_id}</h4>", unsafe_allow_html=True)
                # Detailed Table
                st.dataframe(display_df, hide_index=True)

        # PDF Export Section
        st.markdown("### 📥 Export Report")
        pdf_bytes_cluster = create_cluster_pdf(cluster_export_data, data['x_label'], data['y_label'])
        st.download_button(
            label="Download Cluster Report PDF",
            data=pdf_bytes_cluster,
            file_name="cluster_report.pdf",
            mime="application/pdf"
        )
                        