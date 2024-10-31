import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import urllib3
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession, Row
import plotly.graph_objects as go

# Khởi tạo Spark Session
spark = SparkSession.builder.appName("News Scraper with Sentiment Analysis").getOrCreate()
http = urllib3.PoolManager()
analyzer = SentimentIntensityAnalyzer()
translator = Translator()

# Hàm lấy đoạn introduction
def get_introduction(url):
    r = http.request('GET', url)
    soup = BeautifulSoup(r.data, "html.parser")
    introduction = soup.find("h2", {"class": "intro"})
    return introduction.text.strip() if introduction else "No introduction"

# Hàm lấy các bài báo mới nhất
def get_latest_articles(symbol, limit=10):
    data_rows = []
    url = f'https://s.cafef.vn/Ajax/Events_RelatedNews_New.aspx?symbol={symbol}&floorID=0&configID=0&PageIndex=1&PageSize={limit}&Type=2'
    r = http.request('GET', url)
    soup = BeautifulSoup(r.data, "html.parser")
    data = soup.find("ul", {"class": "News_Title_Link"})
    if not data:
        return []

    raw = data.find_all('li')
    for row in raw:
        news_date = row.span.text.strip()
        title = row.a.text.strip()
        article_url = "https://s.cafef.vn/" + str(row.a['href'])
        introduction = get_introduction(article_url)
        data_rows.append(Row(news_date=news_date, title=title, url=article_url, symbol=symbol, introduction=introduction))
        if len(data_rows) >= limit:
            break
    return data_rows

# Hàm phân tích cảm xúc với VADER
def vader_analyze(row):
    combined_text = f"{row['title_en']} {row['introduction_en']}".strip() if row['introduction_en'] != "No introduction" else row['title_en']
    sentiment_score = analyzer.polarity_scores(combined_text)
    score = (sentiment_score['compound'] + 1) * 50
    return pd.Series([score, "NEGATIVE" if score < 34 else "NEUTRAL" if score < 67 else "POSITIVE"])

# Dashboard Streamlit
st.title("Phân Tích Cảm Xúc Tin Tức Chứng Khoán")

# Nhập mã cổ phiếu
symbol = st.text_input("Nhập mã cổ phiếu:", "SAB")

if st.button("Phân tích"):
    # Lấy dữ liệu tin tức
    news_data = get_latest_articles(symbol, limit=10)
    
    if news_data:
        df_spark_news = spark.createDataFrame(news_data)
        df_pandas_news = df_spark_news.toPandas()
        
        # Dịch tiêu đề và phần giới thiệu sang tiếng Anh
        df_pandas_news['title_en'] = df_pandas_news['title'].apply(lambda x: translator.translate(x, src='vi', dest='en').text)
        df_pandas_news['introduction_en'] = df_pandas_news['introduction'].apply(lambda x: translator.translate(x, src='vi', dest='en').text)
        
        # Áp dụng phân tích cảm xúc và tính điểm
        df_pandas_news[['article_score', 'article_sentiment']] = df_pandas_news.apply(vader_analyze, axis=1)
        
        # Hiển thị bảng tin tức và kết quả
        st.write("### Tin Tức")
        st.dataframe(df_pandas_news[['news_date', 'title', 'introduction', 'article_score', 'article_sentiment']])

        # Tính trung bình 10 bài báo gần nhất
        average_sentiment = df_pandas_news['article_score'].mean()
        
        # Vẽ biểu đồ gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=average_sentiment,
            title={'text': "Đánh Giá Cảm Xúc Tổng Hợp", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 33], 'color': '#FF4C4C'},  # Negative
                    {'range': [33, 66], 'color': '#FFDD57'},  # Neutral
                    {'range': [66, 100], 'color': '#4CAF50'}  # Positive
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': average_sentiment
                }
            }
        ))
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig)

    else:
        st.write("Không có dữ liệu tin tức.")
