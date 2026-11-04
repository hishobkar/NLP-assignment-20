
# Assignment 20: Movie Recommendation System

## Project Overview

This project implements a content-based movie recommendation system
using Python, Pandas, Scikit-learn and Streamlit.

The application recommends movies based on similarities between
their descriptions.

## Dataset

Dataset Name: TMDB 5000 Movie Dataset

Kaggle Link:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

CSV File:
tmdb_5000_movies.csv

Columns used:
- title
- overview

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Cosine Similarity
- Git and GitHub
- Render

## Project Structure

GenAI-Task20-Abhijeet-Hishobkar/
|
|-- app.py
|-- tmdb_5000_movies.csv
|-- requirements.txt
|-- README.md

## How the Recommendation System Works

1. Load the movie dataset using Pandas.
2. Select movie titles and descriptions.
3. Handle missing values.
4. Convert descriptions to lowercase.
5. Remove punctuation and special characters.
6. Remove English stopwords.
7. Apply TF-IDF vectorization.
8. Calculate cosine similarity between movies.
9. Recommend movies with the highest similarity scores.
10. Display recommendations using Streamlit.

## TF-IDF Configuration

- max_features = 5000
- ngram_range = (1, 2)

## How to Run Locally

### Step 1: Install Python

Install Python 3.10 or later.

### Step 2: Install Dependencies

Open a terminal inside the project folder and run:

pip install -r requirements.txt

### Step 3: Run the Application

streamlit run app.py

### Step 4: Open the Application

Streamlit will display a local URL, usually:

http://localhost:8501

## Recommendation Method

This project uses Content-Based Filtering.

TF-IDF converts movie descriptions into numerical vectors.
Cosine similarity measures how similar the movie vectors are.

Movies with higher similarity scores are recommended.

## GitHub Repository

Add your GitHub repository URL here:

https://github.com/hishobkar/NLP-assignment-20

## Render Deployment

Build Command:

pip install -r requirements.txt

Start Command:

streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Deployed Application URL:

Add your Render URL here after successful deployment.

## Conclusion

The project successfully implements a content-based movie
recommendation system. Users can select a movie and receive
recommendations based on the similarity of movie descriptions.
