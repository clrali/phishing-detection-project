# imports needed for web scraping
import requests as re
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

# imports needed for creating the structured data
from bs4 import BeautifulSoup
import pandas as pd
import featureExtraction as fe

disable_warnings(InsecureRequestWarning)

# Step 1: csv to dataframe
phishing_URL_file_name = "datasets/verified_phishing_urls.csv"
legitimate_URL_file_name = "datasets/verified_legitimate_urls.csv"

phishing_data_frame = pd.read_csv(phishing_URL_file_name)
legitimate_data_frame = pd.read_csv(legitimate_URL_file_name)

# retrieve only "url" column and convert it to a list
phishing_URL_list = phishing_data_frame['url'].to_list()
legitimate_URL_list = legitimate_data_frame['url'].to_list()

# restrict the URL count
beginURLIdx = 0
endURLIdx = 500
phishing_collection_list = phishing_URL_list[beginURLIdx:endURLIdx]
legitimate_collection_list = legitimate_URL_list[beginURLIdx:endURLIdx]

# only for the legitimate sites, need to add "http://" prefix
legitimate_collection_list = ["http://" + url for url in legitimate_collection_list]

# function to scrape the content of the URL and convert to a structured form for each
def create_structured_data(url_list):
    data_list = []
    for i in range(0, len(url_list)):
        try:
            response = re.get(url_list[i], verify=False, timeout=1)
            if response.status_code != 200:
                continue
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = fe.create_vector(soup)
                vector.append(str(url_list[i]))
                data_list.append(vector)
                print(i, "processed:", url_list[i])
        except re.exceptions.RequestException as e:
            continue
    return data_list


# phishing_data = create_structured_data(phishing_collection_list)
legitimate_data = create_structured_data(legitimate_collection_list)

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
legitimate_df = pd.DataFrame(data=legitimate_data, columns=columns)

# labeling 0 for legitimate and 1 for phishing
legitimate_df['label'] = 0
# phishing_df['label'] = 1

# save to csv
legitimate_df.to_csv("datasets/structured_legitimate_data.csv", mode='a', index=False, header=False)
# phishing_df.to_csv("datasets/structured_phishing_data.csv", mode='a', index=False, header=False)