import requests
from bs4 import BeautifulSoup
import sys
import json
import time
from urllib.parse import urlparse
import os

def extract_first_page_url(url):
    """
    Extracts the product detail URL from the main catalog list.
    """
    try:
        if not urlparse(url).scheme:
             url = "https://" + url 
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        first_td = soup.find("td", class_="custom__table-heading__title")

        if first_td:
            anchor_tag = first_td.find('a')
            if anchor_tag and 'href' in anchor_tag.attrs:
                return anchor_tag['href']
                
        return None
    
    except Exception as e:
        print(f"Error finding first page URL: {e}")
        return None

def extract_shl_info_to_json(url):
    """
    Fetches SHL product catalog data and extracts specific fields into a structured dictionary.
    """
    try:
        if not urlparse(url).scheme:
             url = "https://" + url 
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to load {url}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Mapping: HTML Header Text -> Desired JSON Key
        field_mapping = {
            "description": "description",
            "job levels": "job_levels",
            "languages": "languages",
            "assessment length": "assessment_length"
        }
        
        # Initialize dictionary with URL and default values
        extracted_data = {
            "url": url,
            "description": "Not found",
            "job_levels": "Not found",
            "languages": "Not found",
            "assessment_length": "Not found"
        }
        
        # Find all rows that match the class structure
        # Note: class_ matches even if the element has multiple classes (e.g., "row typ")
        rows = soup.find_all("div", class_="product-catalogue-training-calendar__row")
        
        for row in rows:
            header_tag = row.find("h4")
            if header_tag:
                header_text = header_tag.get_text(strip=True).lower()
                
                # Check which field this row corresponds to
                for search_term, json_key in field_mapping.items():
                    if search_term in header_text:
                        p_tag = row.find("p")
                        if p_tag:
                            # Clean text and assign to specific key
                            clean_text = p_tag.get_text(separator=" ", strip=True)
                            extracted_data[json_key] = clean_text
                        break

        return extracted_data

    except Exception as e:
        print(f"Error scraping product details: {e}")
        return None

if __name__ == "__main__":
    output_filename = "shl_data_structured.json"
    base_url = "https://www.shl.com"
    
    cells = []
    
    # Optional: Load existing data to resume
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r", encoding="utf-8") as f:
                cells = json.load(f)
            print(f"Resuming... Loaded {len(cells)} existing records.")
        except json.JSONDecodeError:
            print("Could not read existing file. Starting fresh.")
            cells = []

    # Loop through the pages
    for i in range(1, 377):
        print(f"Processing catalog page {i}...")
        
        home_url = f"https://www.shl.com/products/product-catalog/?start={i}&type=1&type=1"
        relative_path = extract_first_page_url(home_url)
        
        if relative_path:
            target_url = base_url + relative_path
            
            # Avoid duplicates
            if any(d.get('url') == target_url for d in cells):
                print(f"Skipping duplicate: {target_url}")
                continue

            data = extract_shl_info_to_json(target_url)
            
            if data:
                cells.append(data)
                print(f"Scraped: {target_url}")

                # --- INCREMENTAL SAVE ---
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(cells, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    print(f"Error saving to file: {e}")

        else:
            print(f"Could not find a product link on catalog page {i}")

        time.sleep(1) # Be polite