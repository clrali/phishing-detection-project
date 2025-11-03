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
# function to scrape the content of the URL and convert to a structured form for each
def create_structured_data(url_list):
    data_list = []
    for i in range(0, len(url_list)):
        try:
            response = re.get(url_list[i], verify=False, timeout=1)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = fe.create_vector(soup)
                vector.append(str(url_list[i]))
                data_list.append(vector)
                print(i, "processed:", url_list[i])
        except re.exceptions.RequestException as e:
            continue
    return data_list

def getResponse(url):
    try:
        response = re.get(url, verify=False, timeout=1)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = fe.create_vector(soup)
            vector.append(url)
            print("processed: ", url)
            return vector
    except re.exceptions.RequestException as e:
        pass
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

    loader = URLLoader(maxURLs = 100)
    cleanTrainingSites, cleanTestSites = loader.getCleanURLs(numURLs = 100, split = 1.0)

    cleanSiteData = pool_get(cleanTrainingSites)

    # phishing_data = create_structured_data(phishing_collection_list)
    # legitimate_data = create_structured_data(legitimate_collection_list)
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
    
    # phishing_df = pd.DataFrame(data=phishing_data, columns=columns)
    legitimate_df = pd.DataFrame(data=cleanSiteData, columns=columns)
    
    # labeling 0 for legitimate and 1 for phishing
    legitimate_df['label'] = 0
    # phishing_df['label'] = 1
    
    # save to csv
    legitimate_df.to_csv("datasets/structured_legitimate_data.csv", mode='a', index=False, header=False)
    # phishing_df.to_csv("datasets/structured_phishing_data.csv", mode='a', index=False, header=False)
