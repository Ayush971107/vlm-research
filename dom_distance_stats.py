import pandas as pd
import numpy as np
import logging
from lxml import html
from zss import Node, simple_distance
from bs4 import BeautifulSoup
import time
import os
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DOM distance functions from scrape3.py
def parse_html_to_dom(html_str):
    try:
        return html.fromstring(html_str)
    except Exception as e:
        logger.error(f"Error parsing HTML to DOM: {str(e)}")
        # Return a minimal DOM if parsing fails
        return html.fromstring("<html><body></body></html>")

def build_zss_tree(dom_node):
    node = Node(dom_node.tag)
    for child in dom_node:
        # Skip text nodes or non-element nodes
        if isinstance(child.tag, str):
            node.addkid(build_zss_tree(child))
    return node

def compute_tree_edit_distance(html1, html2):
    try:
        # Skip computation for non-existent pages
        if html1 == "page does not exist" or html2 == "page does not exist":
            # Return a high distance value to ensure it's recorded
            return 100  # Using an arbitrary high value instead of threshold
            
        dom1 = parse_html_to_dom(html1)
        dom2 = parse_html_to_dom(html2)
        tree1 = build_zss_tree(dom1)
        tree2 = build_zss_tree(dom2)
        return simple_distance(tree1, tree2)
    except Exception as e:
        logger.error(f"Error computing DOM distance: {str(e)}")
        # Return a value to avoid losing data on error
        return 100  # Using an arbitrary high value

def prettify_html(html_content):
    """Prettify HTML content for better parsing"""
    try:
        return BeautifulSoup(html_content, "html.parser").prettify()
    except Exception as e:
        logger.error(f"Error prettifying HTML: {e}")
        return html_content

def calculate_dom_distance_stats():
    """
    Calculate DOM distance statistics from first50.csv
    
    The function reads the first50.csv file, filters out non-HTML files,
    computes DOM distances between page before commit and page after commit,
    and calculates statistics (max, min, average, median).
    """
    # Read the CSV file
    logger.info("Reading first50.csv...")
    try:
        df = pd.read_csv('data_main.csv')
    except Exception as e:
        logger.error(f"Error reading first50.csv: {e}")
        return
    
    logger.info(f"Total records in CSV: {len(df)}")
    
    # Filter for HTML files
    html_files = df[df['file name'].str.endswith('.html', na=False)]
    logger.info(f"HTML files found: {len(html_files)}")
    
    if len(html_files) == 0:
        logger.warning("No HTML files found in the dataset!")
        return
    
    # Calculate DOM distances
    distances = []
    
    for index, row in html_files.iterrows():
        logger.info(f"Processing row {index+1}/{len(html_files)}")
        
        try:
            before_html = row['page before commit']
            after_html = row['page after commit']
            
            # Ensure we have valid HTML
            if pd.isna(before_html) or pd.isna(after_html):
                logger.warning(f"Missing HTML content for row {index}")
                continue
                
            # Prettify HTML content
            before_html = prettify_html(before_html)
            after_html = prettify_html(after_html)
            
            # Calculate DOM distance
            distance = compute_tree_edit_distance(before_html, after_html)
            
            # Store result
            distances.append({
                'file_name': row['file name'],
                'distance': distance
            })
            
            logger.info(f"Distance calculated: {distance}")
            
        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            continue
    
    if not distances:
        logger.warning("No DOM distances could be calculated!")
        return
    
    # Convert to DataFrame for easier analysis
    distances_df = pd.DataFrame(distances)
    
    # Calculate statistics
    stats = {
        'count': len(distances),
        'max': distances_df['distance'].max(),
        'min': distances_df['distance'].min(),
        'mean': distances_df['distance'].mean(),
        'median': distances_df['distance'].median(),
        'std': distances_df['distance'].std(),
        'percentile_25': np.percentile(distances_df['distance'], 25),
        'percentile_75': np.percentile(distances_df['distance'], 75),
        'unique_urls': html_files['url'].nunique()  # Count unique URLs
    }
    
    # Write statistics to file
    with open('dom_distance_stats.txt', 'w') as f:
        f.write("DOM Distance Statistics for HTML Files in first50.csv\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total HTML files analyzed: {stats['count']}\n")
        f.write(f"Number of unique URLs: {stats['unique_urls']}\n\n")
        f.write(f"Maximum DOM distance: {stats['max']}\n")
        f.write(f"Minimum DOM distance: {stats['min']}\n")
        f.write(f"Average DOM distance: {stats['mean']:.2f}\n")
        f.write(f"Median DOM distance: {stats['median']:.2f}\n")
        f.write(f"Standard deviation: {stats['std']:.2f}\n")
        f.write(f"25th percentile: {stats['percentile_25']:.2f}\n")
        f.write(f"75th percentile: {stats['percentile_75']:.2f}\n\n")
        
        # Add top 10 largest distances
        f.write("Top 10 largest DOM distances:\n")
        f.write("-"*50 + "\n")
        top_10 = distances_df.sort_values('distance', ascending=False).head(10)
        for idx, row in top_10.iterrows():
            f.write(f"{row['file_name']}: {row['distance']}\n")
            
    logger.info(f"Statistics written to dom_distance_stats.txt")
    
    # Return statistics for reference
    return stats

if __name__ == "__main__":
    start_time = time.time()
    stats = calculate_dom_distance_stats()
    end_time = time.time()
    
    if stats:
        print("\nDOM Distance Statistics Summary:")
        print(f"Total files analyzed: {stats['count']}")
        print(f"Unique URLs: {stats['unique_urls']}")
        print(f"Max: {stats['max']}")
        print(f"Min: {stats['min']}")
        print(f"Average: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds") 