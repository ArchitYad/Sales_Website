# live_sources.py
import os
import requests
from bs4 import BeautifulSoup


# -----------------------------
# NewsAPI
# -----------------------------
def fetch_news_summary(country="mm", max_articles=5):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return "No NewsAPI key configured."

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": api_key,
        "country": country,
        "pageSize": max_articles
    }

    try:
        r = requests.get(url, params=params, timeout=10).json()
        titles = [a["title"] for a in r.get("articles", [])]
        return " | ".join(titles[:max_articles])
    except Exception as e:
        return f"News fetch error: {e}"


# -----------------------------
# Reddit (NO API KEY)
# -----------------------------
def fetch_social_summary(subreddit="myanmar", limit=5):
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot/"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        titles = []
        for h in soup.find_all("h3"):
            titles.append(h.text)
            if len(titles) >= limit:
                break

        return " | ".join(titles) if titles else "No Reddit trends found."
    except Exception as e:
        return f"Reddit fetch error: {e}"
