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

# Checkpoint file to store progress
CHECKPOINT_FILE = "scraper_checkpoint.pkl"
# Rate limit tracking
RATE_LIMIT_RESET_TIME = None
CONSECUTIVE_RATE_LIMITS = 0
MAX_CONSECUTIVE_LIMITS = 3
EXTENDED_WAIT_TIME = 300  
cache = {}

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
# Fetches 10 commits per website, compares each pair of adjacent commits, 
# and extracts the old and new content.

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
            'page after commit', 'new section'
        ]

        # ───────────────── compare adjacent commits ─────────────────
        for i in range(len(commits) - 1):
            commit1 = commits[i]['sha']
            commit2 = commits[i + 1]['sha']

            try:
                diff_data = compare_commit(api_base_url, commit1, commit2)
                message = diff_data['commits'][0]['commit']['message'] if diff_data.get('commits') else ""

                for file in diff_data.get("files", []):
                    file_name = file['filename']

                    if not file_name.lower().endswith(('.html', '.htm')):
                        continue  # skip non‑HTML files
                    

                    status = file['status']
                    url1 = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit2}/{file_name}"
                    url2 = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit1}/{file_name}"

                    if "patch" in file:           # text diff available
                        old_lines, new_lines = extract_old_new_from_patch(file["patch"])

                        # Fetch file contents before and after change
                        try:
                            resp_before = get_cached(url1)
                            html_before = "page does not exist" if "404: Not Found" in resp_before else resp_before
                            resp_after = get_cached(url2)
                            html_after = resp_after

                            # Prettify for readability
                            try:
                                soup_before = BeautifulSoup(html_before, "html.parser").prettify()
                                soup_after  = BeautifulSoup(html_after,  "html.parser").prettify()
                            except Exception as bs_err:
                                logger.error(f"BeautifulSoup error for {file_name}: {bs_err}")
                                soup_before, soup_after = html_before, html_after

                            df_list.append([
                                api_base_url, message, file_name, status,
                                soup_before, old_lines, soup_after, new_lines
                            ])
                        except Exception as fetch_err:
                            logger.error(f"Error fetching HTML for {file_name}: {fetch_err}")
                            continue
                    else:
                        # no text diff (should be rare for .html)
                        df_list.append([
                            api_base_url, message, file_name, status,
                            "", "", "", ""
                        ])
            except Exception as cmp_err:
                logger.error(f"Error comparing commits {commit1}..{commit2}: {cmp_err}")
                continue

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
        existing_df = pd.read_csv("data1.csv")
        all_data = existing_df.values.tolist()
        column_names = existing_df.columns.tolist()
        logger.info(f"Loaded {len(all_data)} existing records from data1.csv")
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
                df_final.to_csv("data1.csv", index=False)
                logger.info(f"Saved {len(all_data)} records to data1.csv")
            
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