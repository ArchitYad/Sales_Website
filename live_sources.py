# live_sources.py
import requests
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

def fetch_news_summary():
    """Fetch top global retail/economic news headlines."""
    if not NEWS_API_KEY:
        return "NewsAPI key missing."

    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}"
        res = requests.get(url).json()

        headlines = [a["title"] for a in res.get("articles", [])[:5]]
        if not headlines:
            return "No news found."

        return "\n".join([f"- {h}" for h in headlines])
    except:
        return "Error fetching news."


def fetch_social_summary():
    """Fetch latest Reddit trending posts (proxy for consumer sentiment)."""
    try:
        url = "https://www.reddit.com/r/AskReddit/top.json?limit=5&t=day"
        res = requests.get(url, headers={"User-agent": "Mozilla/5.0"}).json()

        titles = [
            i["data"]["title"]
            for i in res["data"]["children"][:5]
        ]
        return "\n".join([f"- {t}" for t in titles])
    except:
        return "Unable to fetch trends."
