#!/usr/bin/env python3
import os
import csv
import requests
from bs4 import BeautifulSoup

URL = "https://www.ibm.com/docs/en/i/7.6.0?topic=extensions-standard-c-library-functions-table-by-name"
OUT_DIR = "../assets/"
OUT_FILE = os.path.join(OUT_DIR, "C.csv")

def scrape_full_table(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; scraper/1.0; +https://example.org/bot)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Find all tables on the page
    tables = soup.find_all("table")

    for table in tables:
        # Try to find header cells 
        thead = table.find("thead")
        if thead:
            header_cells = [th.get_text(strip=True) for th in thead.find_all("th")]
        else:
            first_row = table.find("tr")
            if first_row:
                header_cells = [cell.get_text(strip=True) for cell in first_row.find_all(["th", "td"])]
            else:
                header_cells = []

        header_lower = [h.lower() for h in header_cells]

        # Look for table that has "function" and "include" in the header cells
        if any("function" in h for h in header_lower) and any("include" in h for h in header_lower):
            # Extract all rows from the table 
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                if any(cells):  
                    rows.append(cells)
            return header_cells, rows

    return None, None

def save_to_csv(header, rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"Saved {len(rows)} rows (including header) to {filepath}")


print(f"Scraping table from {URL}")
header, rows = scrape_full_table(URL)

if not header or not rows:
    print("Failed to find the table with 'Function' and 'Include' headers on the page.")

save_to_csv(header, rows, OUT_FILE)