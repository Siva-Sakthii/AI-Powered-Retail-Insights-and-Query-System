import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time  # For rate-limiting

# Set Streamlit page configuration
st.set_page_config(page_title="Sentiment Analyzer with Review Insights", layout="wide")
st.title("Text Document Sentiment Analyzer")
st.header("Analyze the sentiment of a text document and get actionable insights")

# Configure Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Hugging Face model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function for sentiment classification using Hugging Face
def classify_sentiment_hf(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**tokens)
    scores = outputs.logits.detach().numpy()[0]
    scores = softmax(scores)  # Convert logits to probabilities
    sentiment = np.argmax(scores)  # Get the index of the highest probability
    sentiments = ["Negative", "Neutral", "Positive"]
    return sentiments[sentiment], scores[sentiment]

# Function to normalize category names
def normalize_category(category):
    """Cleans and standardizes the category names."""
    if category:
        category = category.strip().lower()
        category = category.lstrip("1234567890. ").rstrip(".")  # Remove leading/trailing numbers and dots
        category = category.replace("-", " ")  # Replace dashes with spaces
        # Standardize names
        if "service" in category:
            return "Service-related issue"
        elif "ambience" in category:
            return "Ambience-related issue"
        elif "product" in category:
            return "Product-related issue"
        elif "price" in category:
            return "Price-related issue"
        else:
            return "Other"
    return "Other"

# Function to classify a single review using Gemini API
def classify_review_with_gemini(review):
    prompt = f"""
    Categorize the following customer review into one of the types:
    - Service-related issue
    - Ambience-related issue
    - Product-related issue
    - Price-related issue
    - Other

    Review: "{review}"
    Return only the category name.
    """
    try:
        model = genai.GenerativeModel("gemini-1.0-pro")
        response = model.generate_content(prompt)
        if response and response.text:
            return normalize_category(response.text.strip())
        else:
            return "Other"
    except Exception as e:
        return "Other"

# Function for batch classification using Gemini API
def batch_classify_reviews(reviews):
    batch_prompt = """
    Categorize each of the following reviews into one of these types:
    - Service-related issue
    - Ambience-related issue
    - Product-related issue
    - Price-related issue
    - Other

    """
    for idx, review in enumerate(reviews):
        batch_prompt += f"{idx+1}. {review}\n"
    batch_prompt += "\nReturn the category for each review in the same order, as a list."

    try:
        model = genai.GenerativeModel("gemini-1.0-pro")
        response = model.generate_content(batch_prompt)
        if response and response.text:
            categories = response.text.strip().split("\n")
            return [normalize_category(cat.strip()) for cat in categories]
        else:
            return ["Other"] * len(reviews)
    except Exception as e:
        return ["Other"] * len(reviews)

# File uploader
uploaded_file = st.file_uploader("Upload a text document", type="txt")

if uploaded_file:
    # Read the text file
    content = uploaded_file.read().decode("utf-8")
    st.subheader("Uploaded Document Content")
    st.text_area("Document Preview", content, height=300)

    # Process the text file into lines or paragraphs
    st.subheader("Sentiment Analysis")
    analysis_choice = st.radio(
        "Analyze by:",
        ("Line", "Paragraph"),
        help="Choose whether to analyze sentiment by each line or each paragraph."
    )

    if analysis_choice == "Line":
        text_segments = content.splitlines()
    else:
        text_segments = content.split("\n\n")  # Treat double line breaks as paragraphs

    results = []
    negative_reviews = []  # Collect negative reviews for batch processing
    
    # Sentiment analysis loop
    for segment in text_segments:
        if segment.strip():  # Skip empty lines
            sentiment, confidence = classify_sentiment_hf(segment)
            
            # Assign 'Nil' for positive or neutral sentiments
            sub_category = "Nil" if sentiment != "Negative" else "Pending"
            
            # Append results
            results.append({
                "Text Segment": segment,
                "Sentiment": sentiment,
                "Confidence": confidence,
                "Sub-Category": sub_category
            })
            
            # Collect negative reviews for further processing
            if sentiment == "Negative":
                negative_reviews.append(segment)

    # Batch process negative reviews for sub-categorization
    if negative_reviews:
        with st.spinner("Processing negative reviews for sub-categories... This may take a few seconds."):
            time.sleep(1)  # Short pause before API calls
            batch_categories = batch_classify_reviews(negative_reviews)
        
        # Reassign normalized categories back to results
        batch_index = 0
        for res in results:
            if res["Sentiment"] == "Negative":
                res["Sub-Category"] = batch_categories[batch_index]
                batch_index += 1

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Display the categorized results
    st.subheader("Sentiment Analysis Results")
    st.write("Below is the full table of analyzed text segments with sentiments and sub-categories:")
    styled_df = results_df.style.format({"Confidence": "{:.2%}"})
    st.dataframe(styled_df, use_container_width=True)

    # Provide a download link for the table
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )

    # Display categorized results
    st.subheader("Categorized Sentiments")
    positive_reviews = results_df[results_df["Sentiment"] == "Positive"]
    negative_reviews_df = results_df[results_df["Sentiment"] == "Negative"]

    st.write("### Positive Reviews")
    st.dataframe(positive_reviews)

    st.write("### Negative Reviews with Sub-Categories")
    st.dataframe(negative_reviews_df)

    # Generate analytics charts
    st.subheader("Sentiment Analysis Charts")
    sentiment_counts = results_df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    if not negative_reviews_df.empty:
        negative_sub_category_counts = negative_reviews_df["Sub-Category"].value_counts()
        st.bar_chart(negative_sub_category_counts)

    # Overall suggestions
    st.subheader("Overall Suggestions")
    if not negative_reviews_df.empty:
        sub_category_counts = negative_reviews_df["Sub-Category"].value_counts()
        if "Service-related issue" in sub_category_counts:
            st.write("- Improve service quality by training staff and ensuring prompt responses to customer needs.")
        if "Ambience-related issue" in sub_category_counts:
            st.write("- Enhance the ambience by improving cleanliness, lighting, or decor.")
        if "Product-related issue" in sub_category_counts:
            st.write("- Focus on product quality by addressing recurring issues and ensuring consistency.")
        if "Price-related issue" in sub_category_counts:
            st.write("- Reevaluate pricing strategies and offer better value for money.")
    st.write("- Regularly collect feedback and act on it to continuously improve.")
