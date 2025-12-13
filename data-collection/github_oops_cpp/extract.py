import requests
import re
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import string
import urllib.parse
from typing import List

simplified_base_search_queries = [
    "extension:cpp class",
    "extension:cpp struct",
    "extension:cpp template",
    "extension:cpp namespace",
    "extension:cpp #include",
    "extension:cpp main()",
    "extension:cpp int main()",
    "extension:cpp void main()",
    "extension:cpp if",
    "extension:cpp for",
    "extension:cpp while",
    "extension:cpp switch",
    "extension:cpp try",
    "extension:cpp catch",
    "extension:cpp throw",
    "extension:cpp new",
    "extension:cpp delete",
    "extension:cpp const",
    "extension:cpp static",
    "extension:cpp virtual",
    "extension:cpp override",
    "extension:cpp public",
    "extension:cpp private",
    "extension:cpp protected",
    "extension:cpp return",
    "extension:cpp typedef",
    "extension:cpp enum",
    "extension:cpp union",
    "extension:cpp auto",
    "extension:cpp nullptr",
    "extension:cpp std::vector",
    "extension:cpp std::string",
    "extension:cpp std::map",
    "extension:cpp std::set",
    "extension:cpp std::shared_ptr",
    "extension:cpp std::unique_ptr",
    "extension:cpp std::thread",
    "extension:cpp std::mutex",
    "extension:cpp std::cout",
    "extension:cpp std::cin"
]

BASE_SEARCH_QUERIES = simplified_base_search_queries

SEARCH_QUERIES = BASE_SEARCH_QUERIES

print(f"Updated BASE_SEARCH_QUERIES with {len(BASE_SEARCH_QUERIES)} simplified queries.")
print(f"Regenerated SEARCH_QUERIES with {len(SEARCH_QUERIES)} queries (no path filter).")

# Redacted for privacy reasons
GITHUB_TOKEN = "GITHUB_PERSONAL_ACCESS_TOKEN_HERE" 

SAVE_DIR = "compilable_cpp_files"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_FILES = 100000
NUM_THREADS = 15

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28"
}

file_count_lock = threading.Lock()
file_count = 0

def is_compilable(filepath: str) -> bool:
    """
    Attempts to compile the C++ file using g++ as a quality check.
    This step is the main filter for valid, standalone C++ code.
    """
    output_path = filepath + ".out"
    try:
        result = subprocess.run(
            ["g++", filepath, "-std=c++17", "-o", output_path, "-w"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15
        )
        
        if os.path.exists(output_path):
            os.remove(output_path)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Compilation utility error: {e}")
        return False


def fetch_and_process_file(raw_url: str) -> bool:
    """
    Fetch a C++ file, perform the compilation check, and save it if successful.
    Returns True if the file count limit was reached during processing.
    """
    global file_count
    
    with file_count_lock:
        if file_count >= MAX_FILES:
            return True 

    try:
        file_resp = requests.get(raw_url, headers=HEADERS, timeout=20)
        
        if file_resp.status_code != 200:
            print(f"Skipping {raw_url}: Status {file_resp.status_code}")
            return False

        content = file_resp.text
        
        temp_filename = os.path.join(SAVE_DIR, f"temp_{threading.get_ident()}.cpp")
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(content)
            
        if is_compilable(temp_filename):
            
            with file_count_lock:
                if file_count >= MAX_FILES:
                    return True
                
                idx = file_count
                file_count += 1
                
                filename = os.path.join(SAVE_DIR, f"file_{idx}.cpp")
                
            os.rename(temp_filename, filename)

            print(f"✅ Saved compilable file: {filename} (Total: {file_count}/{MAX_FILES})")
        else:
            os.remove(temp_filename)

        return False

    except Exception as e:
        print(f"Error fetching/processing {raw_url}: {e}")
        try:
            if os.path.exists(temp_filename):
                 os.remove(temp_filename)
        except:
            pass
        return False


def handle_search_rate_limit(response_headers: dict):
    """Checks for and handles the strict GitHub Search API rate limit."""
    
    if 'X-RateLimit-Remaining' in response_headers:
        remaining = int(response_headers['X-RateLimit-Remaining'])
        
        if remaining < 5:
            reset_time = int(response_headers.get('X-RateLimit-Reset', time.time() + 60))
            sleep_time = max(0, reset_time - time.time()) + 5
            print(f"--- ⚠️ Search Rate Limit Low ({remaining}). Sleeping for {sleep_time:.0f} seconds. ---")
            time.sleep(sleep_time)


def scrape_github_code():
    """Main function to iterate through queries and orchestrate the scraping."""
    
    global file_count
    query_index = 0

    print(f"Starting scraping with {len(SEARCH_QUERIES)} unique search queries.")

    while file_count < MAX_FILES and query_index < len(SEARCH_QUERIES):
        query = SEARCH_QUERIES[query_index]
        page = 1
        print(f"\n--- Using Query {query_index + 1}/{len(SEARCH_QUERIES)}: '{query}' ---")

        while file_count < MAX_FILES and page <= 10:
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://api.github.com/search/code?q={encoded_query}&per_page=100&page={page}&sort=indexed"
            
            try:
                response = requests.get(search_url, headers=HEADERS, timeout=15)
                
                handle_search_rate_limit(response.headers)

                if response.status_code != 200:
                    print(f"Error fetching search results (Status: {response.status_code}). Advancing to next query.")
                    print("Response:", response.text)
                    if response.status_code == 422:
                        print("Query failed (often due to being too broad/specific). Skipping.")
                    break
                
                results = response.json().get("items", [])
                
                if not results:
                    print(f"Page {page} returned no results.")
                    break 
                    
            except Exception as e:
                print(f"Network error fetching search results for query '{query}': {e}. Skipping query.")
                break

            raw_urls = [
                item["html_url"].replace("github.com", "raw.githubusercontent.com").replace("blob/", "")
                for item in results
            ]

            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [executor.submit(fetch_and_process_file, url) for url in raw_urls]
                for future in as_completed(futures):
                    if future.result():
                        print(f"--- Reached target of {MAX_FILES} files. Stopping. ---")
                        return

            page += 1
            time.sleep(1)

        query_index += 1
        
    print(f"\nFinished processing all queries. Total files collected: {file_count}")

if __name__ == "__main__":
    scrape_github_code()