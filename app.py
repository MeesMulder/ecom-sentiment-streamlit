import json
from datetime import datetime
import pandas as pd
import streamlit as st
import altair as alt
from transformers import pipeline

st.set_page_config(page_title="E-commerce Sentiment Monitor (2023)", layout="wide")

DATA_PATH = "data/scraped_data.json"


@st.cache_data
def load_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )




def month_key_to_label(m: int) -> str:
    return datetime(2023, m, 1).strftime("%b 2023")


def main():
    st.title("E-commerce Sentiment Monitor (2023)")
    st.caption("Scraped from web-scraping.dev • Reviews filtered by month • Sentiment via Hugging Face Transformers")

    # Load scraped dataset
    data = load_data(DATA_PATH)

    section = st.sidebar.radio("Navigation", ["Products", "Testimonials", "Reviews"], index=2)

    if section == "Products":
        st.subheader("Products")
        df = pd.DataFrame(data.get("products", []))
        st.dataframe(df, use_container_width=True)
        st.write(f"Total products: {len(df)}")
        return

    if section == "Testimonials":
        st.subheader("Testimonials")
        df = pd.DataFrame(data.get("testimonials", []))
        st.dataframe(df, use_container_width=True)
        st.write(f"Total testimonials: {len(df)}")
        return

    # --- Reviews (core) ---
    st.subheader("Reviews + Sentiment (Core Feature)")

    reviews = data.get("reviews", [])
    df = pd.DataFrame(reviews)

    if df.empty:
        st.warning("No reviews found in data file. Re-run scrape_data.py.")
        return

    # Ensure date parsing
    # date_iso should already exist, but we'll be defensive
    if "date_iso" not in df.columns:
        df["date_iso"] = None

    df["date"] = pd.to_datetime(df["date_iso"], errors="coerce")

    # Month selector for 2023
    month_labels = [month_key_to_label(m) for m in range(1, 13)]
    chosen_label = st.select_slider("Select a month in 2023", options=month_labels, value="May 2023")
    chosen_month = datetime.strptime(chosen_label, "%b %Y").month

    # Filter reviews to the selected month (and year=2023)
    filtered = df[(df["date"].dt.year == 2023) & (df["date"].dt.month == chosen_month)].copy()

    MAX_REVIEWS = 4
    filtered = filtered.sort_values("date_iso").head(MAX_REVIEWS)

    st.write(f"Reviews in **{chosen_label}**: **{len(filtered)}**")

    if filtered.empty:
        st.info("No reviews in this month. Pick another month.")
        return

    run_sa = st.button("Run sentiment analysis for this month")

    if not run_sa:
        st.info("Click the button to run sentiment analysis (saves memory on cloud).")
        st.stop()


    # Sentiment analysis
    model = load_sentiment_model()

    # Batch predictions for speed
    texts = filtered["text"].fillna("").astype(str).tolist()
    preds = model(texts, batch_size=1, truncation=True)


    filtered["sentiment"] = [p["label"] for p in preds]
    filtered["confidence"] = [float(p["score"]) for p in preds]

    


    # Display filtered reviews table
    show_cols = ["date_iso", "rating", "text", "sentiment", "confidence"]
    for c in show_cols:
        if c not in filtered.columns:
            filtered[c] = None

    st.dataframe(filtered[show_cols].sort_values("date_iso"), use_container_width=True)

    # Visualization: counts + avg confidence by sentiment
    summary = (
        filtered.groupby("sentiment")
        .agg(count=("sentiment", "size"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )

    # Make sure both labels exist for consistent chart
    for lab in ["POSITIVE", "NEGATIVE"]:
        if lab not in summary["sentiment"].tolist():
            summary = pd.concat([summary, pd.DataFrame([{"sentiment": lab, "count": 0, "avg_confidence": 0.0}])], ignore_index=True)

    summary["avg_confidence"] = summary["avg_confidence"].fillna(0.0)

    st.markdown("### Sentiment distribution")
    st.caption("Bar height = number of reviews • Text shows average model confidence for that label")

    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", title="Sentiment"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("avg_confidence:Q", title="Avg confidence", format=".3f"),
            ],
        )
    )

    labels = (
        alt.Chart(summary)
        .mark_text(dy=-10)
        .encode(
            x="sentiment:N",
            y="count:Q",
            text=alt.Text("avg_confidence:Q", format=".3f"),
        )
    )

    st.altair_chart(chart + labels, use_container_width=True)


if __name__ == "__main__":
    main()
