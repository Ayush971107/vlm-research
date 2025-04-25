import pandas as pd
import numpy as np
import logging
from lxml import html
from zss import Node, simple_distance
from bs4 import BeautifulSoup
import time
import os
import pickle
import multiprocessing as mp
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_FILE = 'dd_checkpoint.pkl'
CSV_FILE = 'data_main.csv'
PARQUET_FILE = 'data_main_html.parquet'
CHUNK_SIZE = 10  

# Global variable for worker processes
html_df = None

def init_worker(parquet_path):
    global html_df
    # Use fast Parquet reader, loading only required columns
    html_df = pd.read_parquet(parquet_path, columns=[
        'page before commit',
        'page after commit',
        'file name'
    ])
    
def parse_html_to_dom(html_str):
    try:
        return html.fromstring(html_str)
    except Exception as e:
        # Return a minimal DOM if parsing fails
        return html.fromstring("<html><body></body></html>")

def build_zss_tree(dom_node):
    node = Node(dom_node.tag)
    for child in dom_node:
        if isinstance(child.tag, str):
            node.addkid(build_zss_tree(child))
    return node

def compute_tree_edit_distance(html1, html2):
    dom1 = parse_html_to_dom(html1)
    dom2 = parse_html_to_dom(html2)
    tree1 = build_zss_tree(dom1)
    tree2 = build_zss_tree(dom2)
    return simple_distance(tree1, tree2)

def prettify_html(html_content):
    try:
        return BeautifulSoup(html_content, "html.parser").prettify()
    except Exception as e:
        return html_content

def process_row(idx):
    """
    Worker function: given the row index, retrieve the HTML data and compute the distance.
    
    Returns tuple of (idx, distance, file_name, error_type, error_msg)
    """
    try:
        row = html_df.iloc[idx]
        before_html = row['page before commit']
        after_html = row['page after commit']
        file_name = row['file name']
        
        if pd.isna(before_html) or pd.isna(after_html):
            return idx, None, file_name, "missing", "Missing HTML content"

        before_html = prettify_html(before_html)
        after_html = prettify_html(after_html)
        dist = compute_tree_edit_distance(before_html, after_html)
        return idx, dist, file_name, None, None

    except Exception as e:
        error_msg = f"Error processing row {idx}: {str(e)}"
        return idx, None, file_name if 'file_name' in locals() else "unknown", "error", error_msg

def save_checkpoint(current_row, stats):
    """Save checkpoint data to resume processing later"""
    checkpoint = {
        'current_row': current_row,
        'stats': stats
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    logger.info(f"Checkpoint saved at row {current_row}")

def load_checkpoint():
    """Load checkpoint data if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"Checkpoint loaded. Resuming from row {checkpoint['current_row'] + 1}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    
    # Initialize new stats structure
    stats = {
        'non_zero_count': 0,       # Count of distances > 0
        'zero_count': 0,           # Count of distances = 0
        'sum': 0,                  # Sum for mean calculation
        'sum_squared': 0,          # Sum of squares for std calculation
        'min': float('inf'),       # Minimum non-zero distance
        'max': 0,                  # Maximum distance
        'top_10': []               # List of (distance, file_name) tuples for top 10
    }
    return {'current_row': -1, 'stats': stats}

def prepare_parquet_file():
    if os.path.exists(PARQUET_FILE):
        logger.info(f"Parquet file {PARQUET_FILE} already exists, using it")
        return True
    logger.info(f"Converting {CSV_FILE} to Parquet format for faster loading")
    try:
        logger.info(f"Reading {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        html_files = df[df['file name'].str.endswith('.html', na=False)]
        logger.info(f"Found {len(html_files)} HTML files out of {len(df)} total rows")
        html_files.reset_index(drop=True).to_parquet(PARQUET_FILE, index=False)
        logger.info(f"Saved HTML files to {PARQUET_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error creating Parquet file: {e}")
        return False

def calculate_dom_distance_stats():
    """
    Calculate DOM distance statistics from data_main.csv
    
    The function reads the data_main.csv file, filters out non-HTML files,
    computes DOM distances between page before commit and page after commit,
    and calculates statistics (max, min, average, median).
    Only pages with DOM distance >= 1 are included in statistics calculations.
    """
    # First convert to Parquet if needed
    if not prepare_parquet_file():
        logger.error("Failed to prepare Parquet file, falling back to CSV")
    
    # Load checkpoint or initialize stats
    checkpoint = load_checkpoint()
    current_row = checkpoint['current_row']
    stats = checkpoint['stats']
    
    # For calculating median and percentiles, we need to track non-zero distances
    non_zero_distances = []
    
    # Load metadata from the HTML files (don't need all data in the main process)
    try:
        # Just get the file structure to understand the indices
        if os.path.exists(PARQUET_FILE):
            logger.info(f"Loading metadata from {PARQUET_FILE}...")
            # Only load minimal info for the main process
            html_files = pd.read_parquet(PARQUET_FILE, columns=['file name'])
            logger.info(f"Found {len(html_files)} HTML rows in Parquet file")
        else:
            # Fallback to CSV
            logger.info("Reading metadata from data_main.csv...")
            df = pd.read_csv(CSV_FILE, usecols=['file name'])
            html_files = df[df['file name'].str.endswith('.html', na=False)]
            logger.info(f"Found {len(html_files)} HTML files in CSV")
        
        if len(html_files) == 0:
            logger.warning("No HTML files found in the dataset!")
            return
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Get list of row indices to process
    tasks = list(range(current_row + 1, len(html_files)))
    
    if not tasks:
        logger.info("No new rows to process")
        # If there are no tasks but we have stats from previous runs, calculate final stats
        if stats['non_zero_count'] > 0:
            return calculate_final_stats(stats, non_zero_distances, 0)  # URL count will be calculated later
        return None
    
    total_tasks = len(tasks)
    logger.info(f"Processing {total_tasks} rows using multiprocessing")
    
    # Use multiprocessing to parallelize the distance calculations
    num_cores = max(1, mp.cpu_count() - 2)
    logger.info(f"Using {num_cores} CPU cores for parallel processing with chunk size {CHUNK_SIZE}")
    
    # Progress tracking
    completed = 0
    
    # Use multiprocessing pool to process rows in parallel
    with mp.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(PARQUET_FILE,)
    ) as pool:
        for idx, distance, file_name, error_type, error_msg in pool.imap_unordered(process_row, tasks, chunksize=CHUNK_SIZE):
            completed += 1
            
            # Ensure current_row is always the highest processed index
            current_row = max(current_row, idx)
            
            # Handle different result types
            if error_type == "missing":
                logger.warning(f"Missing HTML content for row {idx}")
            elif error_type == "error":
                logger.error(error_msg)
            else:
                # Track zero and non-zero distances separately
                if distance == 0:
                    stats['zero_count'] += 1
                    logger.info(f"Row {idx}: Distance = {distance} (skipped for statistics)")
                else:
                    # Update running statistics for non-zero distances
                    stats['non_zero_count'] += 1
                    stats['sum'] += distance
                    stats['sum_squared'] += distance * distance
                    stats['min'] = min(stats['min'], distance)
                    stats['max'] = max(stats['max'], distance)
                    
                    # Update top 10 list
                    stats['top_10'].append((distance, file_name))
                    stats['top_10'].sort(reverse=True, key=lambda x: x[0])
                    stats['top_10'] = stats['top_10'][:10]  # Keep only top 10
                    
                    # Store for median/percentile calculation
                    non_zero_distances.append(distance)
                    
                    logger.info(f"Row {idx}: Distance = {distance}")
            
            # Log progress (showing completed/total)
            if completed % 10 == 0 or completed == total_tasks:
                logger.info(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
            
            # Save checkpoint every 50 rows instead of after each row
            if completed % 50 == 0:
                logger.info(f"Saving checkpoint at {completed} rows processed")
                save_checkpoint(current_row, stats)
    
    # Final checkpoint after all processing completes
    logger.info("Processing complete. Saving final checkpoint.")
    save_checkpoint(current_row, stats)
    
    logger.info(f"Processed {completed} rows. Calculating final statistics...")
    
    # Now load the full data to get URL count
    url_count = 0
    try:
        # Load to get URL count
        df_for_urls = pd.read_parquet(PARQUET_FILE, columns=['url'])
        url_count = df_for_urls['url'].nunique()
    except Exception as e:
        logger.error(f"Error getting unique URL count: {e}")
    
    return calculate_final_stats(stats, non_zero_distances, url_count)

def calculate_final_stats(stats, non_zero_distances, unique_urls_count):
    """Calculate final statistics from the collected data"""
    # If no non-zero distances were found, return
    if stats['non_zero_count'] == 0:
        logger.warning("No DOM distances >= 1 could be calculated!")
        return
    
    # Calculate final statistics
    mean = stats['sum'] / stats['non_zero_count']
    
    # Calculate standard deviation
    variance = (stats['sum_squared'] / stats['non_zero_count']) - (mean * mean)
    std_dev = max(0, variance) ** 0.5  # Avoid negative values due to floating point errors
    
    # Calculate median and percentiles
    if non_zero_distances:
        median = np.median(non_zero_distances)
        percentile_25 = np.percentile(non_zero_distances, 25)
        percentile_75 = np.percentile(non_zero_distances, 75)
    else:
        # If we're resuming and don't have all distances for accurate percentiles
        logger.warning("Using approximations for median and percentiles (incomplete data)")
        median = mean
        percentile_25 = stats['min']
        percentile_75 = stats['max']
    
    # Prepare final statistics
    final_stats = {
        'count': stats['non_zero_count'],
        'zero_count': stats['zero_count'],
        'max': stats['max'],
        'min': stats['min'],
        'mean': mean,
        'median': median,
        'std': std_dev,
        'percentile_25': percentile_25,
        'percentile_75': percentile_75,
        'unique_urls': unique_urls_count,
        'top_10': stats['top_10']
    }
    
    # Write statistics to file
    with open('dom_distance_stats.txt', 'w') as f:
        f.write("DOM Distance Statistics for HTML Files in data_main.csv\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total HTML files analyzed: {final_stats['count'] + final_stats['zero_count']}\n")
        f.write(f"Files with DOM distance >= 1: {final_stats['count']}\n")
        f.write(f"Files with DOM distance = 0: {final_stats['zero_count']}\n")
        f.write(f"Number of unique URLs: {final_stats['unique_urls']}\n\n")
        f.write(f"Maximum DOM distance: {final_stats['max']}\n")
        f.write(f"Minimum DOM distance: {final_stats['min']}\n")
        f.write(f"Average DOM distance: {final_stats['mean']:.2f}\n")
        f.write(f"Median DOM distance: {final_stats['median']:.2f}\n")
        f.write(f"Standard deviation: {final_stats['std']:.2f}\n")
        f.write(f"25th percentile: {final_stats['percentile_25']:.2f}\n")
        f.write(f"75th percentile: {final_stats['percentile_75']:.2f}\n\n")
        
        # Add top 10 largest distances
        f.write("Top 10 largest DOM distances:\n")
        f.write("-"*50 + "\n")
        for distance, file_name in final_stats['top_10']:
            f.write(f"{file_name}: {distance}\n")
            
    logger.info(f"Statistics written to dom_distance_stats.txt")
    
    # Don't delete the checkpoint file after completion
    # This allows us to avoid reprocessing data when running the script again
    logger.info(f"Checkpoint file '{CHECKPOINT_FILE}' preserved for future runs")
    
    # Return statistics for reference
    return final_stats

if __name__ == "__main__":
    start_time = time.time()
    stats = calculate_dom_distance_stats()
    end_time = time.time()
    
    if stats:
        print("\nDOM Distance Statistics Summary:")
        print(f"Total files analyzed: {stats['count'] + stats['zero_count']}")
        print(f"Files with DOM distance >= 1: {stats['count']}")
        print(f"Files with DOM distance = 0: {stats['zero_count']}")
        print(f"Unique URLs: {stats['unique_urls']}")
        print(f"Max: {stats['max']}")
        print(f"Min: {stats['min']}")
        print(f"Average: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds") 