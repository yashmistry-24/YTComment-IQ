import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to extract video ID from a YouTube URL
def get_video_id(youtube_url):
    video_id_match = re.match(r".*(?:youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=|youtu\.be\/|\/v\/|\/embed\/|v=)([^#\&\?]{11}).*", youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        st.error("Invalid YouTube URL")
        return None

# Function to get all comments from a YouTube video using YouTube Data API
def get_all_comments(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()

    comments = []
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'comment': comment['textDisplay'],
                'date': comment['publishedAt']
            })

        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100,
                textFormat="plainText"
            )
            response = request.execute()
        else:
            break
    
    comments_df = pd.DataFrame(comments)
    return comments_df

# Function to remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Sentiment analysis functions
def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

# LDA Topic Modeling
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topics.append((topic_idx, top_words, topic))
    return topics

# Streamlit Interface
# st.title("YouTube Comments Analysis with Visualizations")
youtube_logo_url = "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg"

# HTML for logo and title
st.markdown(f"""
    <div style='text-align: center;'>
        <img src='{youtube_logo_url}' width='250'>
        <h1>YouTube Comments Analysis with Visualizations</h1>
    </div>
    """, unsafe_allow_html=True)


api_key = 'AIzaSyCdPR3MNnYwqHR2M50e3CTE8UrmwmmFbW4'
youtube_url = st.text_input("Enter a YouTube video URL")

if st.button('Submit'):
    if youtube_url and api_key:
        video_id = get_video_id(youtube_url)
        if video_id:
            with st.spinner("Fetching comments..."):
                comments_df = get_all_comments(video_id, api_key)
                st.success("Comments fetched successfully!")

                # Preprocess comments
                comments_df['cleaned_comment'] = comments_df['comment'].apply(remove_emoji)

                # Sentiment analysis
                comments_df['sentiment_textblob'] = comments_df['cleaned_comment'].apply(textblob_sentiment)
                comments_df['sentiment_vader'] = comments_df['cleaned_comment'].apply(vader_sentiment)

                # Convert TextBlob/VADER sentiment to Positive, Neutral, Negative
                comments_df['sentiment_textblob_label'] = pd.cut(comments_df['sentiment_textblob'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])
                comments_df['sentiment_vader_label'] = pd.cut(comments_df['sentiment_vader'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

                # Add the download button for the CSV file
                csv = comments_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Comments as CSV",
                    data=csv,
                    file_name='youtube_comments.csv',
                    mime='text/csv'
                )

            ### Visualization 1: Sentiment Analysis Comparison: TextBlob vs VADER (Bar Plot)
            st.subheader("Sentiment Analysis Comparison: TextBlob vs VADER")
            textblob_counts = comments_df['sentiment_textblob_label'].value_counts()
            vader_counts = comments_df['sentiment_vader_label'].value_counts()

            sentiment_comparison = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'TextBlob': [textblob_counts.get('Positive', 0), textblob_counts.get('Neutral', 0), textblob_counts.get('Negative', 0)],
                'VADER': [vader_counts.get('Positive', 0), vader_counts.get('Neutral', 0), vader_counts.get('Negative', 0)]
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(x=sentiment_comparison['Sentiment'], y=sentiment_comparison['TextBlob'], name='TextBlob', hoverinfo='y'))
            fig.add_trace(go.Bar(x=sentiment_comparison['Sentiment'], y=sentiment_comparison['VADER'], name='VADER', hoverinfo='y'))

            fig.update_layout(title='Sentiment Analysis Comparison: TextBlob vs VADER',
                              xaxis_title='Sentiment',
                              yaxis_title='Count',
                              barmode='group')
            st.plotly_chart(fig)

            ### Visualization 2: Sentiment Distribution (Pie Chart)
            st.subheader("Sentiment Distribution")
            sentiment_counts = comments_df['sentiment_textblob_label'].value_counts()

            fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=0.4)])
            fig.update_layout(title='Sentiment Distribution')
            st.plotly_chart(fig)

            ### Visualization 3: Top 30 Most Frequent Comments (Horizontal Bar Plot)
            st.subheader("Top 30 Most Frequent Comments")
            comments = comments_df['cleaned_comment'].tolist()
            comment_frequency = Counter(comments)
            most_frequent_comments = comment_frequency.most_common(30)

            # Separate comments and counts
            frequent_comments, counts = zip(*most_frequent_comments)
            df_frequent_comments = pd.DataFrame({
                'comment': frequent_comments,
                'count': counts
            })

            fig = go.Figure(data=[go.Bar(x=df_frequent_comments['count'], y=df_frequent_comments['comment'], orientation='h')])
            fig.update_layout(title='Top 30 Most Frequent Comments',
                              xaxis_title='Number of Comments',
                              yaxis_title='Comments',
                              hovermode='closest')
            st.plotly_chart(fig)

            ### Visualization 4: Sentiment Trend Over Time (Line Plot)
            st.subheader("Sentiment Trend Over Time")
            comments_df['date'] = pd.to_datetime(comments_df['date'])
            sentiment_trend = comments_df.groupby(comments_df['date'].dt.date)['sentiment_vader'].mean().reset_index()

            fig = px.line(sentiment_trend, x='date', y='sentiment_vader', title='Sentiment Trend Over Time')
            fig.update_traces(hovertemplate='Date: %{x}<br>Average Sentiment Score: %{y:.2f}')
            st.plotly_chart(fig)

            ### Visualization 5: Comment Volume Trend Over Time (Line Plot)
            st.subheader("Comment Volume Trend Over Time")
            comment_volume = comments_df.groupby(comments_df['date'].dt.date)['cleaned_comment'].count().reset_index()

            fig = px.line(comment_volume, x='date', y='cleaned_comment', title='Comment Volume Trend Over Time')
            fig.update_traces(hovertemplate='Date: %{x}<br>Number of Comments: %{y}')
            st.plotly_chart(fig)

            ### Visualization 6: WordCloud
            st.subheader("WordCloud of Most Used Words")
            all_comments = " ".join(comments_df['cleaned_comment'].tolist())
            wordcloud = WordCloud(stopwords=set(stopwords.words('english')), background_color='white').generate(all_comments)

            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

            ### Visualization 7: LDA Topic Analysis
            st.subheader("LDA Topic Analysis")

            # Prepare data for LDA
            vectorizer = CountVectorizer(stop_words='english')
            comment_matrix = vectorizer.fit_transform(comments_df['cleaned_comment'])

            lda_model = LDA(n_components=5, random_state=42)
            lda_model.fit(comment_matrix)

            # Get topics and display
            feature_names = vectorizer.get_feature_names_out()
            topics = display_topics(lda_model, feature_names, 5)

            # Display LDA Topics
            for topic_idx, top_words, topic in topics:
                st.write(f"**Topic {topic_idx + 1}:** {top_words}")

                # Create a bar graph for the topic
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_words.split(),
                    y=topic,
                    name=f'Topic {topic_idx + 1}',
                    hoverinfo='y'
                ))
                fig.update_layout(title=f'Top Words for Topic #{topic_idx + 1}',
                                  xaxis_title='Words',
                                  yaxis_title='Weight',
                                  hovermode='closest')
                st.plotly_chart(fig)

