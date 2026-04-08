"""
01b_collect_news.py — Energy News Sentiment Collector
======================================================
Fetches energy-market news from NewsAPI.org, scores each headline with
VADER sentiment analysis, and aggregates into a weekly sentiment CSV.

The CSV accumulates over time — each run appends new weeks without
overwriting historical data.  This lets the model learn from sentiment
as the dataset grows.

Run weekly (ideally right after 01_collect_data.py):
    python scripts/01b_collect_news.py

Outputs: data/raw/news_sentiment.csv
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEWSAPI_KEY = "b4d935087d904f28bf846b06bbfcf2fb"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Search queries targeting Iberian electricity market drivers
# Use AND/OR operators and quotes to stay on-topic and avoid sports/entertainment noise
SEARCH_QUERIES = [
    '"electricity price" AND Europe',
    '"energy market" AND (Spain OR Portugal OR Iberia)',
    '"natural gas" AND (TTF OR price OR Europe)',
    '"carbon" AND (ETS OR EUA OR "emission trading")',
    '"renewable energy" AND (Spain OR Portugal OR Iberia OR generation)',
    '"electricity futures" OR "power futures" OR OMIP OR MIBEL',
    '"European power" AND (market OR price OR wholesale)',
    '(wind OR solar) AND generation AND (Iberia OR Spain OR Portugal)',
    '"hydro" AND (reservoir OR drought) AND (Spain OR Iberia)',
    '"LNG" AND (Europe OR Spain OR terminal)',
]

# Keywords that indicate an article is NOT about energy markets
_IRRELEVANT_KEYWORDS = [
    "world cup", "football", "soccer", "fifa", "copa del rey", "la liga",
    "champions league", "tournament", "olympics", "sport", "match day",
    "baseball", "basketball", "nba", "nfl", "tennis", "formula 1",
    "car review", "truck", "corvette", "vehicle recall",
]

# Languages to include
LANGUAGES = "en,es,pt"

OUTPUT_PATH = config.RAW_DIR / "news_sentiment.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("collect_news")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# VADER Sentiment Analyzer
# ---------------------------------------------------------------------------
_analyzer = SentimentIntensityAnalyzer()

# Add energy-specific terms to boost domain relevance
_energy_lexicon_updates = {
    "bullish": 2.0,
    "bearish": -2.0,
    "surge": 1.5,
    "soar": 1.8,
    "plunge": -1.8,
    "plummet": -2.0,
    "spike": 1.2,
    "crash": -2.5,
    "rally": 1.5,
    "slump": -1.5,
    "shortage": -1.0,
    "surplus": 0.8,
    "drought": -1.5,
    "heatwave": -0.8,
    "coldwave": -0.8,
    "blackout": -2.0,
    "outage": -1.2,
    "record high": 1.5,
    "record low": -1.5,
    "sanctions": -1.0,
    "subsidy": 0.5,
    "renewable": 0.3,
    "nuclear": 0.0,
    "gas crisis": -2.0,
    "energy crisis": -1.8,
    "overcapacity": -0.5,
    "tighten": 0.5,
    "ease": -0.3,
    "volatile": -0.3,
    "stable": 0.3,
}
_analyzer.lexicon.update(_energy_lexicon_updates)


def score_text(text: str) -> dict:
    """Return VADER sentiment scores for a text string."""
    if not text or not isinstance(text, str):
        return {"neg": 0, "neu": 1, "pos": 0, "compound": 0}
    return _analyzer.polarity_scores(text)


# ---------------------------------------------------------------------------
# NewsAPI fetcher
# ---------------------------------------------------------------------------

def _fetch_articles(query: str, from_date: str, to_date: str) -> list[dict]:
    """Fetch articles from NewsAPI for a single query."""
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("articles", [])
        elif resp.status_code == 426:
            logger.warning("NewsAPI free tier limit: %s", resp.json().get("message", ""))
            return []
        else:
            logger.warning("NewsAPI returned %d for query '%s'", resp.status_code, query)
            return []
    except Exception as exc:
        logger.warning("NewsAPI request failed for '%s': %s", query, exc)
        return []


def collect_news_articles(from_date: date, to_date: date) -> pd.DataFrame:
    """Fetch and score all energy news articles in a date range.

    Returns a DataFrame with columns:
        date, source, title, description, url, compound, pos, neg, neu
    """
    from_str = from_date.isoformat()
    to_str = to_date.isoformat()

    logger.info("Fetching news from %s to %s ...", from_str, to_str)

    all_articles = []
    seen_urls = set()

    for query in SEARCH_QUERIES:
        articles = _fetch_articles(query, from_str, to_str)
        for art in articles:
            url = art.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = art.get("title", "") or ""
            description = art.get("description", "") or ""

            # Filter out irrelevant articles (sports, cars, etc.)
            text_lower = f"{title} {description}".lower()
            if any(kw in text_lower for kw in _IRRELEVANT_KEYWORDS):
                continue

            combined_text = f"{title}. {description}"
            scores = score_text(combined_text)

            pub_date = art.get("publishedAt", "")[:10]

            all_articles.append({
                "date": pub_date,
                "source": art.get("source", {}).get("name", ""),
                "title": title,
                "description": description,
                "url": url,
                "compound": scores["compound"],
                "pos": scores["pos"],
                "neg": scores["neg"],
                "neu": scores["neu"],
                "query": query,
            })

    logger.info("Collected %d unique articles across %d queries.",
                len(all_articles), len(SEARCH_QUERIES))

    if not all_articles:
        return pd.DataFrame()

    return pd.DataFrame(all_articles)


def aggregate_weekly_sentiment(articles_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level sentiment into weekly features.

    Returns a weekly DataFrame with:
        news_sentiment   — mean compound score (-1 to +1)
        news_volume      — number of articles
        news_bullish_pct — fraction of articles with compound > 0.05
        news_bearish_pct — fraction of articles with compound < -0.05
        news_max_pos     — strongest positive compound score
        news_max_neg     — most negative compound score
    """
    if articles_df.empty:
        return pd.DataFrame()

    df = articles_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Resample to weekly (Monday)
    df = df.set_index("date").sort_index()

    weekly = df.resample("W-MON").agg(
        news_sentiment=("compound", "mean"),
        news_volume=("compound", "count"),
        news_max_pos=("compound", "max"),
        news_max_neg=("compound", "min"),
    )

    # Bullish/bearish percentages
    def _bullish_pct(group):
        if len(group) == 0:
            return 0
        return (group > 0.05).mean()

    def _bearish_pct(group):
        if len(group) == 0:
            return 0
        return (group < -0.05).mean()

    bull = df["compound"].resample("W-MON").apply(_bullish_pct)
    bear = df["compound"].resample("W-MON").apply(_bearish_pct)
    weekly["news_bullish_pct"] = bull
    weekly["news_bearish_pct"] = bear

    weekly.index.name = "date"
    return weekly


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_and_save() -> None:
    """Main entry: fetch news, score sentiment, save/append to CSV."""
    logger.info("=" * 60)
    logger.info("NEWS SENTIMENT COLLECTION")
    logger.info("=" * 60)

    # NewsAPI free tier: last ~30 days
    to_date = date.today()
    from_date = to_date - timedelta(days=29)

    # Load existing data to avoid duplicating weeks
    existing_weekly = pd.DataFrame()
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_csv(OUTPUT_PATH, comment="#", parse_dates=["date"],
                                   index_col="date")
            existing_weekly = existing
            logger.info("Loaded %d existing weekly rows.", len(existing))
        except Exception:
            pass

    # Fetch new articles
    articles_df = collect_news_articles(from_date, to_date)

    if articles_df.empty:
        logger.warning("No articles fetched. Keeping existing data.")
        return

    # Save raw articles for reference
    articles_path = config.RAW_DIR / "news_articles_latest.csv"
    articles_df.to_csv(articles_path, index=False)
    logger.info("Saved %d raw articles to %s", len(articles_df), articles_path.name)

    # Aggregate to weekly
    new_weekly = aggregate_weekly_sentiment(articles_df)

    if new_weekly.empty:
        logger.warning("No weekly sentiment computed.")
        return

    # Merge with existing (new data overwrites overlapping weeks)
    if not existing_weekly.empty:
        combined = pd.concat([existing_weekly, new_weekly])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = new_weekly.sort_index()

    # Save
    header = f"# Source: NewsAPI.org + VADER sentiment | Updated: {datetime.now().isoformat()}\n"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(header)
        combined.to_csv(fh)

    logger.info("Saved %d weekly sentiment rows to %s", len(combined), OUTPUT_PATH.name)

    # Print summary
    print("\n" + "=" * 60)
    print("NEWS SENTIMENT SUMMARY")
    print("=" * 60)
    print(f"Articles fetched:     {len(articles_df)}")
    print(f"Weekly rows saved:    {len(combined)}")
    print(f"Date range:           {combined.index.min().date()} to {combined.index.max().date()}")
    print(f"Latest sentiment:     {combined['news_sentiment'].iloc[-1]:.3f}")
    print(f"Latest volume:        {int(combined['news_volume'].iloc[-1])} articles")
    print(f"Latest bullish %:     {combined['news_bullish_pct'].iloc[-1]:.1%}")
    print(f"Latest bearish %:     {combined['news_bearish_pct'].iloc[-1]:.1%}")
    print()


if __name__ == "__main__":
    collect_and_save()
