# import streamlit as st
# import pandas as pd

# import function
# import import_ipynb
# import Tweets
# from ntscraper import Nitter
# import neattext.functions as nfx
# from PIL import Image

# def Analyze_Tweets(data):
#     data['Tweets_en'] = data['Tweets'].apply(function.translate_text)
#     data['Tweets_en'] = data['Tweets_en'].apply(function.cleanTxt)
#     data['Tweets_en'] = data['Tweets_en'].apply(nfx.remove_stopwords)
#     data['Tweets_en'] = data['Tweets_en'].apply(nfx.remove_punctuations)
    
#     data['Subjectivity'] = data['Tweets_en'].apply(function.getSubjectivity)
#     data['Polarity'] = data['Tweets_en'].apply(function.getPolarity)
#     data['Analysis'] = data['Polarity'].apply(function.getAnalysis)
#     return data

# # Define the Streamlit app
# # Set title and description
# logo_url1 = Image.open('twitter (2).png')
# logo_url2 = Image.open('sl_z_072523_61700_01.jpg')
# st.image([logo_url1,logo_url2],width = 100)
# st.title("Tweet Analysis Web App")
# st.write("This web app analyzes tweets using the ntscraper model.")

# # Input section
# name = st.text_input("Enter the username: ")
# modes = st.text_input("Enter the mode as term, hashtag or user:")
# no = st.number_input("Enter the number of tweets to analyze: ")
# data = Tweets.get_tweets(name,modes,no)
    
# # Button to trigger analysis
# if st.button("Get Tweets"):
#     # Display predictions
#     st.write("Data:")
#     st.dataframe(data)

# clean_data = Analyze_Tweets(data)
# show_data = clean_data[['Tweets','Analysis']]
    
# col1, col2 = st.columns(2)
# with col1:
#     button1 = st.button("Analyze Data")
# with col2:
#     button2 = st.button("Visualize Data")
        
# if button1:
#     st.write("Predicted Data:")
#     st.dataframe(show_data)
    
# if button2:
#     viz1, viz2 = st.columns((2,2))
#     with viz1:
#         sample_text = data['Tweets'].tolist()
#         word_cloud = function.generate_wordcloud(sample_text)
#         st.write("WordCLoud:")
#         st.image(word_cloud)
#     with viz2:
#         bar_chart = function.generate_bar_chart(clean_data)
#         st.write("Bar Chart:")
#         st.plotly_chart(bar_chart)

import streamlit as st
import pandas as pd
from PIL import Image
import function
import Tweets
from ntscraper import Nitter
import neattext.functions as nfx
import plotly.graph_objects as go

# Function to analyze tweets
def analyze_tweets(data):
    data['Tweets_en'] = data['Tweets'].apply(function.translate_text)
    data['Tweets_en'] = data['Tweets_en'].apply(function.cleanTxt)
    data['Tweets_en'] = data['Tweets_en'].apply(nfx.remove_stopwords)
    data['Tweets_en'] = data['Tweets_en'].apply(nfx.remove_punctuations)

    data['Language'] = data['Tweets'].apply(function.detect_language)
    data['Subjectivity'] = data['Tweets_en'].apply(function.getSubjectivity)
    data['Polarity'] = data['Tweets_en'].apply(function.getPolarity)
    data['Analysis'] = data['Polarity'].apply(function.getEmoji)
    data['Predictions'] = function.predict_with_model(data['Tweets_en'])
    return data

# Streamlit app main function
def main():
    st.set_page_config(page_title="Tweet Analysis Web App", page_icon=":bar_chart:", layout="wide")

    # Title and description
    st.title("Tweet Analysis Web App")
    st.write("This web app analyzes tweets using various NLP techniques.")
    st.markdown("---")

    # Sidebar for search criteria
    st.sidebar.image('Nitter_logo.png', width=80)
    st.sidebar.title("Search Criteria:")
    name = st.sidebar.text_input("Username:")
    modes = st.sidebar.selectbox("Mode:", ["term", "hashtag", "user"])
    no = st.sidebar.number_input("Number of tweets to analyze:", min_value=1, step=1, value=10)
    get_tweets_button = st.sidebar.button("Get Tweets")

    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = None
        st.session_state.clean_data = None

    # Display logos in the main interface
    col1, col2 = st.columns([1, 4])
    with col1:
        logo1 = Image.open('twitter (2).png')
        logo2 = Image.open('sl_z_072523_61700_01.jpg')
        st.image([logo1, logo2], width=200)

    # Retrieve tweets
    if get_tweets_button:
        st.session_state.data = Tweets.get_tweets(name, modes, no)
        if st.session_state.data is not None:
            st.success("Tweets retrieved successfully!")
            st.dataframe(st.session_state.data)
        else:
            st.error("Failed to retrieve tweets!")

    # Analyze data
    if st.session_state.data is not None:
        st.header("Analyze Data")
        st.write("Click the button to analyze the retrieved tweets.")
        analyze_button = st.button("Analyze Data")
        if analyze_button:
            with st.spinner('Analyzing data...'):
                st.session_state.clean_data = analyze_tweets(st.session_state.data)
            st.success("Data analyzed successfully!")
            st.dataframe(st.session_state.clean_data[['Tweets', 'Predictions', 'Analysis']])

    # Visualize data
    if st.session_state.clean_data is not None:
        st.header("Visualize Data")
        st.write("Click the button to visualize the analyzed data.")
        visualize_button = st.button("Visualize Data")
        if visualize_button:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("Word Cloud:")
                function.generate_wordcloud(st.session_state.clean_data["Tweets"].tolist())
                word_cloud_bar = function.generate_wordcloud_barchart(st.session_state.clean_data["Tweets"].tolist())
                st.plotly_chart(word_cloud_bar)
                likes_count = function.generate_likes_scatter_plot(st.session_state.clean_data)
                st.plotly_chart(likes_count)
            
            with col2:
                donut_chart = function.create_donut_chart(st.session_state.clean_data)
                st.plotly_chart(donut_chart)
                comment_count = function.generate_Comment_plot(st.session_state.clean_data)
                st.plotly_chart(comment_count)
                bar_chart = function.generate_bar_chart(st.session_state.clean_data)
                st.plotly_chart(bar_chart)

if __name__ == "__main__":
    main()


