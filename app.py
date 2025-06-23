import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import re
import string
import uuid
import plotly.graph_objects as go

# File to store corrections
CORRECTIONS_FILE = "corrections.json"

# Initialize session state for corrections and thresholds
if 'corrections' not in st.session_state:
    st.session_state.corrections = {}
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.85
if 'group_id_counter' not in st.session_state:
    st.session_state.group_id_counter = 0

# Function to normalize product names
def normalize_product_name(text):
    if not isinstance(text, str):
        return ""
    
    # Product-specific stop words
    product_stop_words = {
        'model', 'color', 'size', 'dual', 'sim', 'wifi', 'enabled', 'mens', 'womens', 
        'men', 'women', 'type', 'variant', 'version', 'inch', 'gb', 'tb', 'esim'
    }
    
    # Normalize text
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into tokens and remove stop words
    tokens = text.split()
    tokens = [token for token in tokens if token not in product_stop_words]
    
    # Return joined string
    return ' '.join(tokens)

# Function to load corrections
def load_corrections():
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save corrections
def save_corrections(corrections):
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(corrections, f, indent=4)

# Function to visualize product groups with Plotly
def visualize_product_groups(df, name_column, groups):
    # Initialize lists for nodes and edges
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_text = []
    node_labels = []
    node_colors = []
    
    # Generate positions using a simple grid-like layout for clarity
    np.random.seed(42)  # For reproducibility
    n_groups = len(groups)
    group_centers = [(np.cos(2 * np.pi * i / n_groups), np.sin(2 * np.pi * i / n_groups)) for i in range(n_groups)]
    
    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Process each group
    for idx, (group_id, indices) in enumerate(groups.items()):
        # Get center for the group
        center_x, center_y = group_centers[idx % len(group_centers)]
        n_products = len(indices)
        
        # Arrange products in a circle around the center
        for i, product_idx in enumerate(indices):
            product = df.iloc[product_idx][name_column]
            short_label = ' '.join(product.split()[:3])  # First 3 words for label
            angle = 2 * np.pi * i / max(n_products, 1)
            radius = 0.3
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            node_x.append(x)
            node_y.append(y)
            node_text.append(product)  # Full name for hover
            node_labels.append(short_label)
            node_colors.append(colors[idx % len(colors)])
            
            # Add edges between all products in the group
            for j in range(i + 1, len(indices)):
                other_idx = indices[j]
                other_x = center_x + radius * np.cos(2 * np.pi * j / max(n_products, 1))
                other_y = center_y + radius * np.sin(2 * np.pi * j / max(n_products, 1))
                edge_x.extend([x, other_x, None])
                edge_y.extend([y, other_y, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition='top center',
        textfont=dict(size=10, color='black'),
        marker=dict(
            showscale=False,
            color=node_colors,
            size=20,
            line_width=1
        ),
        hovertext=node_text
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Product Grouping Map',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Function to deduplicate products
def deduplicate_products(df, name_column, threshold):
    # Normalize product names
    product_names = df[name_column].apply(normalize_product_name).tolist()
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(product_names)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Load existing corrections
    corrections = load_corrections()
    
    # Initialize groups
    groups = {}
    visited = set()
    
    for i in range(len(product_names)):
        if i in visited:
            continue
        group_id = str(uuid.uuid4())
        groups[group_id] = [i]
        visited.add(i)
        
        for j in range(i + 1, len(product_names)):
            if j in visited:
                continue
            # Check if there's a correction
            product_pair = tuple(sorted([product_names[i], product_names[j]]))
            correction_key = f"{product_pair[0]}|{product_pair[1]}"
            if correction_key in corrections:
                if corrections[correction_key]:
                    groups[group_id].append(j)
                    visited.add(j)
            else:
                if similarity_matrix[i, j] >= threshold:
                    groups[group_id].append(j)
                    visited.add(j)
    
    return groups

# Streamlit app
st.title("Product Deduplication App")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file with product data", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Read Excel file with explicit engine
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        
        # Display column selection
        columns = df.columns.tolist()
        name_column = st.selectbox("Select the column with product names", columns)
        
        # Similarity threshold slider
        st.session_state.similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05
        )
        
        if st.button("Deduplicate Products"):
            # Perform deduplication
            groups = deduplicate_products(df, name_column, st.session_state.similarity_threshold)
            
            # Store groups in session state
            st.session_state.groups = groups
            st.session_state.df = df
            st.session_state.name_column = name_column
            
            # Display groups
            st.subheader("Product Groups")
            for group_id, indices in groups.items():
                st.write(f"**Group {group_id}**")
                group_data = df.iloc[indices][[name_column]].reset_index()
                st.dataframe(group_data)
                
                # Allow user to correct groups
                st.write("Correct this group:")
                products = df.iloc[indices][name_column].tolist()
                for i, product in enumerate(products):
                    if st.button(f"Remove '{product}' from Group {group_id}", key=f"remove_{group_id}_{i}"):
                        # Remove product from group
                        groups[group_id].remove(indices[i])
                        if not groups[group_id]:  # If group is empty, delete it
                            del groups[group_id]
                        else:
                            # Create new group for removed product
                            new_group_id = str(uuid.uuid4())
                            groups[new_group_id] = [indices[i]]
                        
                        # Save correction
                        for other_idx in indices:
                            if other_idx != indices[i]:
                                product_pair = tuple(sorted([
                                    normalize_product_name(df.iloc[indices[i]][name_column]),
                                    normalize_product_name(df.iloc[other_idx][name_column])
                                ]))
                                correction_key = f"{product_pair[0]}|{product_pair[1]}"
                                st.session_state.corrections[correction_key] = False
                                save_corrections(st.session_state.corrections)
                        
                        # Update session state
                        st.session_state.groups = groups
                        st.rerun()
            
            # Display visualization
            st.subheader("Product Grouping Map")
            visualize_product_groups(df, name_column, groups)
        
        # Allow merging groups
        if 'groups' in st.session_state:
            st.subheader("Merge Groups")
            group_ids = list(st.session_state.groups.keys())
            if len(group_ids) > 1:
                group1 = st.selectbox("Select first group to merge", group_ids, key="group1")
                group2 = st.selectbox("Select second group to merge", group_ids, key="group2")
                if st.button("Merge Selected Groups"):
                    if group1 != group2:
                        # Merge groups
                        st.session_state.groups[group1].extend(st.session_state.groups[group2])
                        del st.session_state.groups[group2]
                        
                        # Save corrections for merged products
                        for idx1 in st.session_state.groups[group1]:
                            for idx2 in st.session_state.groups[group1]:
                                if idx1 < idx2:
                                    product_pair = tuple(sorted([
                                        normalize_product_name(df.iloc[idx1][name_column]),
                                        normalize_product_name(df.iloc[idx2][name_column])
                                    ]))
                                    correction_key = f"{product_pair[0]}|{product_pair[1]}"
                                    st.session_state.corrections[correction_key] = True
                                    save_corrections(st.session_state.corrections)
                        
                        # Update session state
                        st.session_state.groups = st.session_state.groups
                        st.rerun()
            
            # Display visualization after merge
            st.subheader("Product Grouping Map")
            visualize_product_groups(st.session_state.df, st.session_state.name_column, st.session_state.groups)
            
            # Download corrected groups
            if st.session_state.groups:
                output_df = pd.DataFrame({
                    'Product Name': df[name_column],
                    'Group ID': ''
                })
                for group_id, indices in st.session_state.groups.items():
                    output_df.loc[indices, 'Group ID'] = group_id
                
                # Convert to Excel
                output_df.to_excel("deduplicated_products.xlsx", index=False)
                with open("deduplicated_products.xlsx", "rb") as f:
                    st.download_button(
                        label="Download Deduplicated Products",
                        data=f,
                        file_name="deduplicated_products.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}. Please ensure the file is a valid .xlsx file and try again.")

# Display current corrections (for debugging)
if st.checkbox("Show current corrections"):
    st.write(st.session_state.corrections)
