# Product Deduplication App

This is a Streamlit-based application for deduplicating products based on their names. It groups similar products (e.g., “IPhone 15 Red 256GB” and “Apple IPhone Model 15 Color Red Size 256GB”) using TF-IDF and cosine similarity, allows manual corrections, and learns from those corrections to improve future groupings.

## Features
- Upload an Excel file with product names.
- Select the column containing product names.
- Adjust the similarity threshold for grouping.
- View and correct groups by removing or merging products.
- Download deduplicated results as an Excel file.
- Learn from corrections using a local JSON file (`corrections.json`).

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`

## Files
- `app.py`: Main Streamlit application code.
- `requirements.txt`: Python dependencies.
- `sample_products.xlsx`: Sample data for testing.
- `README.md`: This file.

## Deployment on Streamlit Cloud
1. **Create a GitHub Repository**:
   - Create a new repository on GitHub.
   - Push `app.py`, `requirements.txt`, and `sample_products.xlsx` to the repository.
2. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
   - Click "New app" and connect to your GitHub repository.
   - Select the repository and branch.
   - Specify `app.py` as the main file.
   - Deploy the app.
3. **Test the App**:
   - Upload `sample_products.xlsx` to test deduplication.
   - Select “Product Name” as the column.
   - Adjust the similarity threshold (default: 0.85).
   - Correct groups and download results.

## Sample Data
The `sample_products.xlsx` file contains 20 products across electronics, clothing, and furniture, with variations in naming to test deduplication. Examples:
- “IPhone 15 Red 256GB ESIM” and “Apple IPhone Model 15 Color Red Size 256GB Dual Sim Wifi”
- “Nike Air Max Blue Mens Size 10” and “Air Max Nike Blue Size 10 Men”

## Usage
1. Upload an Excel file with a column of product names.
2. Select the product name column.
3. Adjust the similarity threshold if needed.
4. Click “Deduplicate Products” to view groups.
5. Correct groups by:
   - Removing products (creates a new group for the removed product).
   - Merging groups (combines two groups).
6. Download the deduplicated results as an Excel file.
7. Corrections are saved to `corrections.json` and applied to future deduplications.

## Notes
- The app uses a local JSON file for persistence. If Streamlit Cloud’s file system is reset, corrections may be lost.
- The similarity threshold controls grouping strictness (higher = stricter).
- For large datasets, deduplication may be slow due to pairwise similarity calculations.

## License
MIT License