# imports needed for web scraping
import requests as re
from ReadDB import URLLoader
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

# imports needed for creating the structured data
from bs4 import BeautifulSoup
import pandas as pd
import featureExtraction as fe

# imports for multiprocessing requests
import time
import asyncio
import aiohttp
from multiprocessing import Pool


disable_warnings(InsecureRequestWarning)

def getResponse(url):
    try:
        response = re.get(url, verify=False, timeout=1)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "lxml")
            features = fe.create_vector(soup)
            features.append(url)
            print("processed: ", url)
            return features
        else:
            return None
    except Exception as e:
        return None

def pool_get(urls):
    resp = []
    with Pool() as p:
        iter = p.imap(getResponse, urls)
        for response in iter:
            if response != None:
                resp.append(response)
    return resp
if __name__ == '__main__':
    numURLs = 50000
    loader = URLLoader(maxURLs = numURLs)
    start = time.time()
    cleanTrainingSites = loader.getCleanURLs(numURLs = numURLs)
    cleanSiteData = pool_get(cleanTrainingSites)

    #phishingSites = loader.getPhishingURLs(numURLs = numURLs)
    #phishingSiteData = pool_get(phishingSites)
    end = time.time()
    print(end - start)

    columns = [
        'has_title',
        'has_input',
        'has_button',
        'has_image',
        'has_submit',
        'has_link',
        'has_password',
        'has_email_input',
        'has_hidden_element',
        'has_audio',
        'has_video',
        'number_of_inputs',
        'number_of_buttons',
        'number_of_images',
        'number_of_option',
        'number_of_list',
        'number_of_th',
        'number_of_tr',
        'number_of_href',
        'number_of_paragraph',
        'number_of_script',
        'length_of_title',
        'has_h1',
        'has_h2',
        'has_h3',
        'length_of_text',
        'number_of_clickable_button',
        'number_of_a',
        'number_of_img',
        'number_of_div',
        'number_of_figure',
        'has_footer',
        'has_form',
        'has_text_area',
        'has_iframe',
        'has_text_input',
        'number_of_meta',
        'has_nav',
        'has_object',
        'has_picture',
        'number_of_sources',
        'number_of_span',
        'number_of_table',
        'URL'
    ]

    #phishing_df = pd.DataFrame(data=phishingSiteData, columns=columns)
    legitimate_df = pd.DataFrame(data=cleanSiteData, columns=columns)

    # labeling 0 for legitimate and 1 for phishing
    legitimate_df['label'] = 0
    #phishing_df['label'] = 1

    # save to csv
    legitimate_df.to_csv("datasets/structured_legitimate_data.csv", mode='w', index=False, header=False)
    #phishing_df.to_csv("datasets/structured_phishing_data.csv", mode='w', index=False, header=False)