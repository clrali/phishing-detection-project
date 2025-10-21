import streamlit as st
import machineLearning as ml
import featureExtraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt

st.title('Phishing Website Detection using Machine Learning')

with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used _supervised learning_ to classify phishing and legitimate URLs by using a content-based approach and focusing on' \
    'various features extracted from the HTML of the web pages. The ML models were implemented using the scikit-learn library.')
    st.write('For this educational project, '
             'I created my own data set and defined features, some from the literature and some based on manual analysis. '
             'I used requests library to collect data, BeautifulSoup module to parse and extract features. ')

    st.subheader('Data set')
    st.write('I used _"phishtank.org"_ to collect verified phishing URLs & _"tranco-list.eu"_ to collect verified legitimate URLs as data sources.')
    st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')
    st.write('Data set was created in October 2025.')

    # # ----- FOR THE PIE CHART ----- #
    # labels = 'phishing', 'legitimate'
    # phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    # legitimate_rate = 100 - phishing_rate
    # sizes = [phishing_rate, legitimate_rate]
    # explode = (0.1, 0)
    # fig, ax = plt.subplots()
    # ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    # ax.axis('equal')
    # st.pyplot(fig)
    # # ----- !!!!! ----- #

    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))


    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    st.subheader('Features')
    st.write('I used only content-based features. I didn\'t use url-based faetures like length of url etc.'
             'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

    st.subheader('Results')
    st.write('I used 7 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
             'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
             'Comparison table is below:')
    st.table(ml.df_results)
    st.write('NB --> Gaussian Naive Bayes')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbours')

choice = st.selectbox("Please select your machine learning model",
                 [
                     'Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                     'AdaBoost', 'Neural Network', 'K-Neighbours'
                 ]
                )

if choice == 'Gaussian Naive Bayes':
    model = ml.models['NB']
    st.write('GNB model is selected!')
elif choice == 'Support Vector Machine':
    model = ml.models['SVM']
    st.write('SVM model is selected!')
elif choice == 'Decision Tree':
    model = ml.models['DT']
    st.write('DT model is selected!')
elif choice == 'Random Forest':
    model = ml.models['RF']
    st.write('RF model is selected!')
elif choice == 'AdaBoost':
    model = ml.models['AB']
    st.write('AB model is selected!')
elif choice == 'Neural Network':
    model = ml.models['NN']
    st.write('NN model is selected!')
else:
    model = ml.models['KN']
    st.write('KN model is selected!')

url = st.text_input('Enter the URL')
# check the url is valid or not
if st.button('Check!'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            print(". HTTP connection was not successful for the URL: ", url)
            st.warning("The HTTP connection was not successful for this URL!")
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
            result = model.predict(vector)
            if result[0] == 0:
                st.success("This web page seems a legitimate!")
            else:
                st.warning("Attention! This web page is a potential PHISHING!")

    except re.exceptions.RequestException as e:
        print("--> ", e)