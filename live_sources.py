import requests
import os

# -------------------------
# 1️⃣ NEWS API (Market, Retail, Economy)
# -------------------------
def fetch_news_summary():
    api_key = os.getenv("NEWSAPI_KEY", "YOUR_NEWS_API_KEY")

    url = (
        "https://newsapi.org/v2/top-headlines?"
        "category=business&"
        "language=en&"
        f"apiKey={api_key}"
    )

    try:
        res = requests.get(url)
        data = res.json()

        if "articles" not in data:
            return "No major economic headlines available."

        headlines = [a["title"] for a in data["articles"][:5]]
        return " • " + "\n • ".join(headlines)

    except:
        return "Could not fetch economic news."


# -------------------------
# 2️⃣ Social Media (Twitter/X or Reddit)
#    Using simple RSS gateway (no OAuth required)
# -------------------------
def fetch_social_summary():
    try:
        # Reddit topic feed: r/economy / retail / consumer behavior
        feed_url = "https://www.reddit.com/r/retail/.rss"
        resp = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})

        if resp.status_code != 200:
            return "No social trend data found."

        # extract titles from RSS
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)

        titles = []
        for item in root.iter("{http://www.w3.org/2005/Atom}title"):
            titles.append(item.text)

        return " • " + "\n • ".join(titles[:5])

    except:
        return "Could not fetch social media trend signals."
