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
import seaborn as sns
import matplotlib.pyplot as plt
import umap

# Set matplotlib backend for Streamlit
plt.switch_backend('Agg')

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
    
    # Lowercase, remove punctuation, collapse spaces
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into tokens and remove stop words
    tokens = text.split()
    tokens = [token for token in tokens if token not in product_stop_words]
    
    # Join tokens back into string
    return ' '.join(tokens)

# Function to load corrections from file
def load_corrections():
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save corrections to file
def save_corrections(corrections):
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(corrections, f, indent=4)

# Function to visualize groups
def visualize_groups(df, name_column, groups):
    # Normalize product names
    product_names = df[name_column].apply(normalize_product_name).tolist()
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(product_names)
    
    # Reduce dimensions to 2D using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(tfidf_matrix)
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'Product': df[name_column],
        'Group': ''
    })
    
    # Assign group IDs
    for group_id, indices in groups.items():
        plot_data.loc[indices, 'Group'] = group_id
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_data,
        x='x',
        y='y',
        hue='Group',
        palette='deep',
        s=100,
        legend=False
    )
    
    # Add labels for a few products (avoid clutter)
    for i, row in plot_data.iterrows():
        if i % 3 == 0:  # Label every 3rd product to avoid overcrowding
            plt.text(row['x'] + 0.1, row['y'], row['Product'][:20], fontsize=8)
    
    plt.title("Product Groups Visualization")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    # Display plot in Streamlit
    st.pyplot(plt)

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
            st.subheader("Visualization of Product Groups")
            visualize_groups(df, name_column, groups)
        
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
            st.subheader("Visualization of Product Groups")
            visualize_groups(st.session_state.df, st.session_state.name_column, st.session_state.groups)
            
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
