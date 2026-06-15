import requests
import json
import os

def download_euroncap_json(url, filename="euroncap_data.json"):
    # It is crucial to include a User-Agent header, 
    # otherwise the website may block the request.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.euroncap.com/en/results/"
    }

    try:
        print(f"Connecting to: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        
        # Raise an exception if the request was unsuccessful (e.g., 404, 500, 403)
        response.raise_for_status()

        # Parse JSON data
        data = response.json()

        # Save to a file with pretty printing
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully downloaded and saved to: {os.path.abspath(filename)}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    # Your specific URL
    # Note: To get 'All' cars, the path is often just '%2Fresults'
    api_url = "https://www.euroncap.com/api/CarListRoute?path=%2Fassessments%2F1228"
    
    download_euroncap_json(api_url, "euroncap_results.json")