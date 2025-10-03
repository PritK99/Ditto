import time
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Setup for Selenium WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Set up the base URL for the index page
BASE_URL = "https://www.ibm.com/docs/en/zos/3.2.0?topic=reference-standard-c-library-header-files"

# Function to fetch and parse page using Selenium
def get_selenium_soup(url):
    driver.get(url)
    
    # Wait for the page to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))  # Wait until the body is loaded
    time.sleep(2)  # Additional sleep for any dynamic content
    page_source = driver.page_source
    return BeautifulSoup(page_source, 'html.parser')

# Function to scrape and extract relevant content from header page
def extract_relevant_sections(header_url):
    soup = get_selenium_soup(header_url)
    sections = {"functions": [], "classes": [], "templates": [], "objects": [], "macros": []}

    # Find all headers (e.g., <h2>, <h3>, etc.) to identify sections like functions, classes, etc.
    headings = soup.find_all(['h2', 'h3', 'h4'])

    for heading in headings:
        heading_text = heading.get_text(strip=True).lower()

        # Skip 'Description' and 'Synopsis'
        if 'description' in heading_text or 'synopsis' in heading_text:
            continue

        # Extract specific sections
        if 'function' in heading_text:
            # Collect functions under this section
            next_section = heading.find_next_sibling()
            if next_section:
                functions = extract_items(next_section)
                sections['functions'].extend(functions)
        
        elif 'class' in heading_text:
            next_section = heading.find_next_sibling()
            if next_section:
                classes = extract_items(next_section)
                sections['classes'].extend(classes)

        elif 'template' in heading_text:
            next_section = heading.find_next_sibling()
            if next_section:
                templates = extract_items(next_section)
                sections['templates'].extend(templates)

        elif 'object' in heading_text:
            next_section = heading.find_next_sibling()
            if next_section:
                objects = extract_items(next_section)
                sections['objects'].extend(objects)

        elif 'macro' in heading_text:
            next_section = heading.find_next_sibling()
            if next_section:
                macros = extract_items(next_section)
                sections['macros'].extend(macros)

    return sections

# Function to extract items (like functions, classes, templates) from sections
def extract_items(section):
    items = []
    if section.name == 'ul':
        # Extract items from unordered list (<ul>)
        for li in section.find_all('li'):
            item = li.get_text(strip=True)
            if item:
                items.append(item)
    elif section.name == 'table':
        # Extract items from a table
        for row in section.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) > 0:
                item = cols[0].get_text(strip=True)
                if item:
                    items.append(item)
    return items

# Function to scrape the index page and get all header links
def scrape_index_page():
    soup = get_selenium_soup(BASE_URL)
    header_links = {}

    # Find all links to header pages
    for a in soup.find_all('a', href=True):
        link_text = a.get_text(strip=True)
        href = a['href']

        # Check if the link matches the expected pattern for header files
        if href and re.match(r'/docs/en/zos/3.2.0\?topic=files-[a-zA-Z0-9_-]+', href):
            header_links[link_text] = 'https://www.ibm.com' + href

    return header_links

# Function to save the scraped data to a JSON file
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Main function to scrape all headers and save the results
def main():
    # Scrape the index page for all header links
    header_links = scrape_index_page()
    print(f"Found {len(header_links)} header links.")
    
    header_data = {}
    
    # Loop through all header links and scrape their content
    for header_name, header_url in header_links.items():
        try:
            print(f"Scraping {header_name}: {header_url}")
            sections = extract_relevant_sections(header_url)
            header_data[header_name] = sections
            time.sleep(1)  # Sleep to avoid too many requests in a short time
        except Exception as e:
            print(f"Error scraping {header_name}: {e}")
    
    # Save the scraped data to a JSON file
    save_to_json(header_data, "cpp_header_data.json")
    print("Data saved to cpp_header_data.json")

if __name__ == "__main__":
    main()
