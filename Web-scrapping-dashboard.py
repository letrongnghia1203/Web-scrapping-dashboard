import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import urllib3
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# Initialize HTTP PoolManager and sentiment analyzer
http = urllib3.PoolManager()
analyzer = SentimentIntensityAnalyzer()

# Function to fetch introduction from a website
def get_introduction(url):
    r = http.request('GET', url)
    soup = BeautifulSoup(r.data, "html.parser")
    introduction = soup.find("h2", {"class": "intro"})
    return introduction.text.strip() if introduction else "No introduction"

# Function to fetch latest articles
def get_latest_articles(symbol, limit=10):
    data_rows = []
    url = f'https://s.cafef.vn/Ajax/Events_RelatedNews_New.aspx?symbol={symbol}&floorID=0&configID=0&PageIndex=1&PageSize={limit}&Type=2'
    r = http.request('GET', url)
    soup = BeautifulSoup(r.data, "html.parser")
    data = soup.find("ul", {"class": "News_Title_Link"})
    if not data:
        return pd.DataFrame()  # Return empty DataFrame if no data

    raw = data.find_all('li')
    for row in raw:
        news_date = row.span.text.strip()
        title = row.a.text.strip()
        article_url = "https://s.cafef.vn/" + str(row.a['href'])
        introduction = get_introduction(article_url)
        data_rows.append({"news_date": news_date, "title": title, "url": article_url, "symbol": symbol, "introduction": introduction})
        if len(data_rows) >= limit:
            break
    return pd.DataFrame(data_rows)

# Function to translate text from Vietnamese to English
def translate_text(text):
    try:
        return GoogleTranslator(source='vi', target='en').translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text

# VADER sentiment analysis function
def vader_analyze(row):
    combined_text = f"{row['title_en']} {row['introduction_en']}".strip() if row['introduction_en'] != "No introduction" else row['title_en']
    sentiment_score = analyzer.polarity_scores(combined_text)
    score = (sentiment_score['compound'] + 1) * 50
    return pd.Series([score, "NEGATIVE" if score < 34 else "NEUTRAL" if score < 67 else "POSITIVE"])

# Streamlit dashboard
st.title("Phân Tích Cảm Xúc Tin Tức Chứng Khoán")

# User input for stock symbol
symbol = st.text_input("Nhập mã cổ phiếu:")

if symbol and st.button("Phân tích"):
    # Fetch news data
    df_pandas_news = get_latest_articles(symbol, limit=10)
    
    if not df_pandas_news.empty:
        # Translate title and introduction to English
        df_pandas_news['title_en'] = df_pandas_news['title'].apply(translate_text)
        df_pandas_news['introduction_en'] = df_pandas_news['introduction'].apply(translate_text)
        
        # Apply sentiment analysis and calculate score
        df_pandas_news[['article_score', 'article_sentiment']] = df_pandas_news.apply(vader_analyze, axis=1)

        # Display news table with clickable links
        st.write("### Tin Tức")
        for index, row in df_pandas_news.iterrows():
            st.markdown(f"**Ngày**: {row['news_date']} | **Tiêu đề**: [{row['title']}]({row['url']}) | **Cảm xúc**: {row['article_sentiment']} | **Điểm**: {row['article_score']:.2f}")

        # Average sentiment score of the 10 latest articles
        average_sentiment = df_pandas_news['article_score'].mean()
        
        # Gauge chart for overall sentiment score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=average_sentiment,
            title={'text': "Đánh Giá Cảm Xúc Tổng Hợp", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 33], 'color': '#FF4C4C'},
                    {'range': [33, 66], 'color': '#FFDD57'},
                    {'range': [66, 100], 'color': '#4CAF50'}
                ],
                'threshold': {
                    'line': {'color': "black"},
                    'thickness': 0.75,
                    'value': average_sentiment
                }
            }
        ))
        st.plotly_chart(fig)

        # Line chart of article scores over time
        st.write("### Điểm Cảm Xúc Theo Thời Gian")
        df_pandas_news['news_date'] = pd.to_datetime(df_pandas_news['news_date'])
        df_pandas_news = df_pandas_news.sort_values(by='news_date')
        st.line_chart(df_pandas_news.set_index('news_date')['article_score'])

        # Bar chart of sentiment counts
        st.write("### Phân Bổ Số Lượng Bài Viết Theo Cảm Xúc")
        sentiment_counts = df_pandas_news['article_sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        # Text preprocessing for word cloud
        df_pandas_news['cleaned_title_en'] = df_pandas_news['title_en'].str.replace(r'\W', ' ')
        df_pandas_news['cleaned_introduction_en'] = df_pandas_news['introduction_en'].str.replace(r'\W', ' ')
        
        text_data = df_pandas_news['cleaned_title_en'] + " " + df_pandas_news['cleaned_introduction_en']
        
        # TF-IDF matrix and KMeans clustering
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        df_pandas_news['cluster_id'] = kmeans.fit_predict(tfidf_matrix)
        
        # Word cloud for the densest cluster
        densest_cluster_id = df_pandas_news['cluster_id'].value_counts().idxmax()
        densest_cluster_text = " ".join(
            df_pandas_news[df_pandas_news['cluster_id'] == densest_cluster_id]['cleaned_title_en'] + " " +
            df_pandas_news[df_pandas_news['cluster_id'] == densest_cluster_id]['cleaned_introduction_en']
        )
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(densest_cluster_text)
        
        # Display Word Cloud
        st.write(f"### Word Cloud for Densest Cluster (Cluster ID: {densest_cluster_id})")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("Không có dữ liệu tin tức.")
