from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import time
import os
import logging
import hashlib
import pickle
import datetime
from zss import Node, simple_distance
from lxml import html

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for GitHub API
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else None
}

# Threshold for DOM distance - only include changes with distance >= threshold
DOM_DISTANCE_THRESHOLD = 30

# Checkpoint file to store progress
CHECKPOINT_FILE = "scraper_checkpoint.pkl"
# Rate limit tracking
RATE_LIMIT_RESET_TIME = None
CONSECUTIVE_RATE_LIMITS = 0
MAX_CONSECUTIVE_LIMITS = 3
EXTENDED_WAIT_TIME = 300  
cache = {}

# Step 1: Parse HTML DOM
def parse_html_to_dom(html_str):
    try:
        return html.fromstring(html_str)
    except Exception as e:
        logger.error(f"Error parsing HTML to DOM: {str(e)}")
        # Return a minimal DOM if parsing fails
        return html.fromstring("<html><body></body></html>")

# Step 2: Convert lxml DOM to zss.Node tree
def build_zss_tree(dom_node):
    node = Node(dom_node.tag)
    for child in dom_node:
        # Skip text nodes or non-element nodes
        if isinstance(child.tag, str):
            node.addkid(build_zss_tree(child))
    return node

# Step 3: Compute Tree Edit Distance
def compute_tree_edit_distance(html1, html2):
    try:
        # Skip computation for non-existent pages
        if html1 == "page does not exist" or html2 == "page does not exist":
            # Return a high distance value to ensure it's recorded
            return DOM_DISTANCE_THRESHOLD + 10
            
        dom1 = parse_html_to_dom(html1)
        dom2 = parse_html_to_dom(html2)
        tree1 = build_zss_tree(dom1)
        tree2 = build_zss_tree(dom2)
        return simple_distance(tree1, tree2)
    except Exception as e:
        logger.error(f"Error computing DOM distance: {str(e)}")
        # Return the threshold value to avoid losing data on error
        return DOM_DISTANCE_THRESHOLD

# Creates unique cache key from the URL and headers, returns cached response if available
def get_cached(url, headers=None):
    global RATE_LIMIT_RESET_TIME, CONSECUTIVE_RATE_LIMITS
    
    # Create a cache key from the URL and headers
    headers_str = str(sorted(headers.items())) if headers else ""
    cache_key = f"{url}:{headers_str}"
    
    # Use a hash for very long keys
    if len(cache_key) > 100:
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
    
    # Return from cache if available
    if cache_key in cache:
        return cache[cache_key]
    
    # Make the request
    try:
        response = requests.get(url, headers=headers)
        
        # Handle GitHub API rate limiting
        if response.status_code == 403 and 'API rate limit exceeded' in response.text:
            CONSECUTIVE_RATE_LIMITS += 1
            
            # Try to get reset time from headers
            reset_time = None
            if 'X-RateLimit-Reset' in response.headers:
                reset_timestamp = int(response.headers['X-RateLimit-Reset'])
                reset_time = datetime.datetime.fromtimestamp(reset_timestamp)
                RATE_LIMIT_RESET_TIME = reset_time
                
                # Calculate how long to wait
                now = datetime.datetime.now()
                wait_seconds = (reset_time - now).total_seconds() + 5  # Add 5 seconds buffer
                
                if wait_seconds > 0:
                    logger.warning(f"Rate limit hit ({CONSECUTIVE_RATE_LIMITS} times). Reset at {reset_time}. Waiting {int(wait_seconds)} seconds...")
                    time.sleep(wait_seconds)
                    CONSECUTIVE_RATE_LIMITS = 0
                    return get_cached(url, headers)
            
            # If we can't get reset time or hit limits multiple times
            if CONSECUTIVE_RATE_LIMITS >= MAX_CONSECUTIVE_LIMITS:
                logger.warning(f"Too many consecutive rate limits. Taking an extended break ({EXTENDED_WAIT_TIME} seconds)...")
                time.sleep(EXTENDED_WAIT_TIME)
                CONSECUTIVE_RATE_LIMITS = 0
            else:
                logger.warning(f"Rate limit hit for {url}, waiting 60 seconds...")
                time.sleep(60)
            
            return get_cached(url, headers)
            
        CONSECUTIVE_RATE_LIMITS = 0  # Reset counter on successful request
        
        # Store in cache and return
        result = response.text
        cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Error in get_cached for {url}: {str(e)}")
        raise

# Parse a Git patch to extract the old and new content.
def extract_old_new_from_patch(patch_text):
    old_version = []
    new_version = []

    for line in patch_text.split("\n"):
        if line.startswith("---") or line.startswith("+++"):  # Ignore file headers
            continue
        elif line.startswith("-") and not line.startswith("--"):
            old_version.append(line[1:].strip())
        elif line.startswith("+") and not line.startswith("++"):
            new_version.append(line[1:].strip())

    return old_version, new_version

def compare_commit(api_base_url, commit1, commit2):
    url = f"{api_base_url}/compare/{commit2}...{commit1}"
    response_text = get_cached(url, headers=HEADERS)
    return json.loads(response_text)

# Fetches commits and only records HTML diffs when DOM distance exceeds threshold
def scrape_website(owner, repo):
    try:
        # ───────────────── repo info ─────────────────
        api_base_url = f"https://api.github.com/repos/{owner}/{repo}"
        response_text = get_cached(api_base_url, headers=HEADERS)
        repo_data = json.loads(response_text)

        # Rate‑limit check
        if 'message' in repo_data and 'API rate limit exceeded' in repo_data['message']:
            logger.error(f"Rate limit exceeded for {owner}/{repo}")
            return [], []

        branch = repo_data.get("default_branch", "main")

        # ───────────────── latest commits ─────────────────
        url = f"{api_base_url}/commits?sha={branch}"
        response_text = get_cached(url, headers=HEADERS)
        commits = json.loads(response_text)

        if not isinstance(commits, list) or len(commits) < 2:
            logger.warning(f"Not enough commits for {owner}/{repo}")
            return [], []

        commits = commits[:10]  # limit to 10 most‑recent commits

        # ───────────────── data collection ─────────────────
        df_list = []
        column_names = [
            'url', 'commit message', 'file name', 'file status',
            'page before commit', 'section modified',
            'page after commit', 'new section', 
            'dom_distance', 'first_commit_sha', 'last_commit_sha'
        ]

        # Track HTML files across commits
        html_files_tracked = {}  # filename -> {latest_content, first_commit, latest_commit, etc}

        # ───────────────── process commits ─────────────────
        # First pass: collect all HTML files and their changes across commits
        for i in range(len(commits) - 1):
            commit1 = commits[i]['sha']
            commit2 = commits[i + 1]['sha']
            
            try:
                diff_data = compare_commit(api_base_url, commit1, commit2)
                message = diff_data['commits'][0]['commit']['message'] if diff_data.get('commits') else ""

                for file in diff_data.get("files", []):
                    file_name = file['filename']

                    # Skip non-HTML files
                    if not file_name.lower().endswith(('.html', '.htm')):
                        continue

                    status = file['status']
                    url1 = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit2}/{file_name}"
                    url2 = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit1}/{file_name}"

                    # Process files with patches (text diffs)
                    if "patch" in file:
                        old_lines, new_lines = extract_old_new_from_patch(file["patch"])

                        try:
                            # Get content before and after change
                            resp_before = get_cached(url1)
                            html_before = "page does not exist" if "404: Not Found" in resp_before else resp_before
                            resp_after = get_cached(url2)
                            html_after = resp_after

                            # Prettify for readability
                            try:
                                soup_before = BeautifulSoup(html_before, "html.parser").prettify()
                                soup_after = BeautifulSoup(html_after, "html.parser").prettify()
                            except Exception as bs_err:
                                logger.error(f"BeautifulSoup error for {file_name}: {bs_err}")
                                soup_before, soup_after = html_before, html_after

                            # If this is the first time we've seen this file
                            if file_name not in html_files_tracked:
                                html_files_tracked[file_name] = {
                                    'first_content': soup_before,
                                    'latest_content': soup_after,
                                    'first_commit': commit2,
                                    'latest_commit': commit1,
                                    'latest_message': message,
                                    'latest_status': status,
                                    'latest_old_lines': old_lines,
                                    'latest_new_lines': new_lines
                                }
                            else:
                                # We've seen this file before, update with latest content
                                current = html_files_tracked[file_name]
                                
                                # Calculate DOM distance between first and latest content
                                dom_distance = compute_tree_edit_distance(
                                    current['first_content'], 
                                    soup_after
                                )
                                
                                # If distance exceeds threshold, record this diff and reset tracking
                                if dom_distance >= DOM_DISTANCE_THRESHOLD:
                                    df_list.append([
                                        api_base_url, 
                                        current['latest_message'], 
                                        file_name, 
                                        current['latest_status'],
                                        current['first_content'], 
                                        current['latest_old_lines'],
                                        soup_after, 
                                        current['latest_new_lines'],
                                        dom_distance,
                                        current['first_commit'],
                                        commit1  # latest commit
                                    ])
                                    
                                    # Reset tracking with current content as new baseline
                                    html_files_tracked[file_name] = {
                                        'first_content': soup_after,
                                        'latest_content': soup_after,
                                        'first_commit': commit1,
                                        'latest_commit': commit1,
                                        'latest_message': message,
                                        'latest_status': status,
                                        'latest_old_lines': old_lines,
                                        'latest_new_lines': new_lines
                                    }
                                else:
                                    # Update the latest state without recording
                                    current['latest_content'] = soup_after
                                    current['latest_commit'] = commit1
                                    current['latest_message'] = message
                                    current['latest_status'] = status
                                    current['latest_old_lines'] = old_lines
                                    current['latest_new_lines'] = new_lines
                                    
                        except Exception as fetch_err:
                            logger.error(f"Error fetching HTML for {file_name}: {fetch_err}")
                            continue
            except Exception as cmp_err:
                logger.error(f"Error comparing commits {commit1}..{commit2}: {cmp_err}")
                continue

        # Second pass: record any remaining tracked files that didn't exceed threshold
        # (this ensures every repo has at least one entry if it has HTML files)
        for file_name, data in html_files_tracked.items():
            # Skip files already recorded
            if data['first_commit'] == data['latest_commit']:
                continue
                
            # Calculate final DOM distance
            dom_distance = compute_tree_edit_distance(
                data['first_content'], 
                data['latest_content']
            )
            
            # Record the diff between first and latest commits
            df_list.append([
                api_base_url, 
                data['latest_message'], 
                file_name, 
                data['latest_status'],
                data['first_content'], 
                data['latest_old_lines'],
                data['latest_content'], 
                data['latest_new_lines'],
                dom_distance,
                data['first_commit'],
                data['latest_commit']
            ])

        return df_list, column_names

    except Exception as top_err:
        logger.error(f"Error scraping {owner}/{repo}: {top_err}")
        return [], []


def save_checkpoint(index):
    """Save current index to a checkpoint file"""
    checkpoint = {
        'index': index,
        'timestamp': datetime.datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    logger.info(f"Saved checkpoint at index {index}")

def load_checkpoint():
    """Load index from checkpoint file if it exists"""
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
        
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        logger.info(f"Loaded checkpoint from {checkpoint['timestamp']} at index {checkpoint['index']}")
        return checkpoint['index']
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return 0

def main():
    # Load checkpoint if available
    start_index = load_checkpoint()
    
    # Read the existing websites file
    try:
        df = pd.read_csv('github_websites.csv')
    except FileNotFoundError:
        logger.error("github_websites.csv not found")
        return
    
    if start_index >= len(df):
        logger.info("All websites have been processed. Reset checkpoint to start over.")
        return
        
    logger.info(f"Starting from index {start_index}")
    
    # Load existing data if available
    try:
        existing_df = pd.read_csv("data3.csv")
        all_data = existing_df.values.tolist()
        column_names = existing_df.columns.tolist()
        logger.info(f"Loaded {len(all_data)} existing records from data3.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        all_data = []
        column_names = None
        logger.info("No existing data found, starting fresh")
    
    try:
        # Process each repository in the list, starting from the checkpoint
        for index, row in df.iloc[start_index:].iterrows():
            owner = row['Owner']
            repo = row['Repository']
            
            logger.info(f"Scraping {owner}/{repo} ({index+1}/{len(df)})")
            df_list, cols = scrape_website(owner, repo)
            
            # Skip if no data was found for this repository
            if not df_list:
                logger.warning(f"No data found for {owner}/{repo}")
                # Still save checkpoint so we don't retry this repo
                save_checkpoint(index + 1)
                continue
                
            # Set column names if this is the first successful scrape
            if column_names is None and cols:
                column_names = cols
                
            # Add the records to our main dataset
            all_data.extend(df_list)
            
            # Save progress after each website to avoid losing data
            if all_data and column_names:
                df_final = pd.DataFrame(all_data, columns=column_names)
                df_final.to_csv("data3.csv", index=False)
                logger.info(f"Saved {len(all_data)} records to data3.csv")
            
            # Update checkpoint with just the index
            save_checkpoint(index + 1)
                
            # Rate limiting between repositories
            time.sleep(2)
            
    # Handle user interruption (Ctrl+C)
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Progress is saved.")
            
    # Handle unexpected errors
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.info("Progress checkpoint is saved, you can resume from this point.")

def reset_checkpoint():
    """Utility function to reset the checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Checkpoint reset. Next run will start from the beginning.")
    else:
        logger.info("No checkpoint file found.")

if __name__ == "__main__":
    main()
