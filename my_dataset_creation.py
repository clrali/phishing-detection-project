# datasetBuilder.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
from my_features import extract_features_from_html
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

disable_warnings(InsecureRequestWarning)

class SiteCollector:
    def __init__(self, timeout=2):
        self.timeout = timeout

    def fetch(self, url: str):
        """Return HTML text for URL or None if request fails."""
        try:
            resp = requests.get(url, timeout=self.timeout, verify=False)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        return None

    def process_urls(self, urls):
        """Convert list of URLs into feature vectors."""
        rows = []
        for idx, url in enumerate(urls):
            html = self.fetch(url)
            if html is None:
                continue

            features = extract_features_from_html(html)
            features.append(url)  # add the URL
            rows.append(features)
            print(f"[{idx}] processed: {url}")

        return rows


def save_dataset(rows, filename, feature_names):
    df = pd.DataFrame(rows, columns=feature_names)
    df.to_csv(filename, index=False)
    print(f"\nSaved dataset → {filename}")


if __name__ == "__main__":
    # EXAMPLE ONLY — Replace these with our URL lists
    example_urls = [
        "https://www.google.com",
        "https://www.mit.edu",
        "https://www.reddit.com",
    ]

    # define 43 feature names + URL at the end
    feature_columns = [f"f{i}" for i in range(43)] + ["URL"]

    collector = SiteCollector(timeout=2)
    rows = collector.process_urls(example_urls)

    save_dataset(rows, "datasets/my_legitimate_sites.csv", feature_columns)
