# YTComment-IQ

YTComment-IQ is a web-based application built to analyze and visualize YouTube comments from any video. Powered by natural language processing techniques, the app offers sentiment analysis, topic modeling, word frequency insights, and various visualizations to help you gain meaningful insights from YouTube comments.

[Enjoy YTComment-IQ](https://ytcomment-iq.streamlit.app/)

## Features

### 1. Sentiment Analysis:

- Comparison between two sentiment analysis techniques: TextBlob and VADER.

- Sentiment distribution visualization using a pie chart.

### 2. Comment Volume and Sentiment Trends:
- Visualize trends in sentiment over time with a line graph.
- Track the volume of comments over time to identify spikes and activity levels.

### 3. WordCloud Visualization:
- Displays the most frequent words from the comments as a word cloud, excluding common stopwords.

### 4. LDA Topic Modeling:
- Automatically extract and visualize dominant topics in comments using Latent Dirichlet Allocation (LDA).

### 5. Top 30 Most Frequent Comments:
- A bar chart visualizing the top 30 most frequently occurring comments.

### 6. Downloadable Reports:
- Export analyzed comments and results as a CSV file for further use.

## How It Works
1. **Enter YouTube Video URL:** Users simply input a YouTube video URL. The app will extract the video ID and begin fetching the comments using the YouTube Data API.

2. **Analyze Comments:** The app cleans and processes the comments by removing emojis and analyzing sentiments using both TextBlob and VADER. It also groups the comments by date to provide a temporal view of sentiment and volume trends.

3. **Visualize Data:** The results are presented through interactive visualizations, including:

- Bar plots for sentiment comparison and topic words.
- Pie charts for sentiment distribution.
- Line charts for trends over time.
- Word clouds for frequently used words.

4. **LDA Topic Modeling:** The app performs LDA to detect hidden topics in the comment data, showing the most relevant words for each identified topic.

5. **Export Results:** Users can download the analyzed comment data as a CSV file for further exploration or reporting.

## User Interface (UI)
The application has an intuitive interface with the following sections:

1. Input Area: Users enter the video URL. An "Enter" button triggers the analysis.

2. Visualizations: Various charts display sentiment analysis comparisons, trends over time, most frequent comments, and LDA topics.

3. Download Button: Users can download the analyzed data as a CSV file.

The UI is enhanced with interactive features like tooltips that display data values when hovering over graphs.

## How to Use

1. Paste the YouTube video URL in the input field.
2. Click Enter to start the analysis.
3. Once the comments are fetched and analyzed, the results are displayed with various visualizations.
4. You can download the comment data as a CSV file using the download button.

## Contribution

### Get Started

1. Clone this repository:

```bash
git clone https://github.com/username/YTComment-IQ.git
```

2. Navigate to the project directory:

```bash
cd YTComment-IQ
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Dependencies
- Streamlit: For building the web app interface.
- YouTube Data API: For fetching comments from YouTube videos.
- TextBlob & VADER: For performing sentiment analysis on the comments.
- Plotly: For interactive visualizations.
- WordCloud: For generating word clouds.
- sklearn: For LDA topic modeling.
- Matplotlib: For additional visual plots.


## License
This project is licensed under the MIT License.