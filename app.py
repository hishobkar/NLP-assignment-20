
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# TASK 1: LOAD AND UNDERSTAND DATASET
# --------------------------------------------------

DATASET_PATH = "tmdb_5000_movies.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDATASET INFORMATION")
print("-------------------")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Keep only the columns needed for recommendations
df = df[["title", "overview"]].copy()

# Handle missing values
df["title"] = df["title"].fillna("")
df["overview"] = df["overview"].fillna("")

# Remove rows without a movie title
df = df[df["title"].str.strip() != ""]

# Remove duplicate movie titles
df = df.drop_duplicates(subset=["title"])

df = df.reset_index(drop=True)

print("\nDataset after basic preparation:", df.shape)


# --------------------------------------------------
# TASK 2: TEXT PREPROCESSING
# --------------------------------------------------

stop_words = set(ENGLISH_STOP_WORDS)


def clean_text(text):

    # Convert text into lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Create a new cleaned text column
df["clean_text"] = df["overview"].apply(clean_text)

print("\nTEXT PREPROCESSING")
print("------------------")

print(df[["title", "overview", "clean_text"]].head())


# --------------------------------------------------
# TASK 3: TF-IDF VECTORIZATION
# --------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(df["clean_text"])

print("\nTF-IDF VECTORIZATION")
print("--------------------")

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

print("Number of Features:", len(tfidf.get_feature_names_out()))


# --------------------------------------------------
# TASK 4: COSINE SIMILARITY
# --------------------------------------------------

# Store similarity values in a sparse matrix
similarity_matrix = cosine_similarity(
    tfidf_matrix,
    dense_output=False
)

print("\nCOSINE SIMILARITY")
print("-----------------")

print("Similarity Matrix Shape:", similarity_matrix.shape)

print("Similarity Matrix Type:", type(similarity_matrix))


# --------------------------------------------------
# TASK 5: RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend(item_name, top_n=5):

    # Find the selected movie
    matching_movies = df[
        df["title"].str.lower() == item_name.lower()
    ]

    if matching_movies.empty:
        return pd.DataFrame(
            columns=["title", "similarity_score"]
        )

    # Get the index of the selected movie
    movie_index = matching_movies.index[0]

    # Get similarity scores for the selected movie
    scores = similarity_matrix.getrow(movie_index).toarray().flatten()

    # Sort movies by similarity score
    similar_indices = scores.argsort()[::-1]

    # Exclude the selected movie itself
    similar_indices = [
        index for index in similar_indices
        if index != movie_index
    ]

    # Select top N recommendations
    top_indices = similar_indices[:top_n]

    recommendations = df.iloc[top_indices][["title"]].copy()

    recommendations["similarity_score"] = [
        round(float(scores[index]), 4)
        for index in top_indices
    ]

    return recommendations.reset_index(drop=True)


# --------------------------------------------------
# TEST RECOMMENDATION FUNCTION WITH 3 MOVIES
# --------------------------------------------------

print("\nRECOMMENDATION TESTING")
print("----------------------")

test_movies = [
    "Avatar",
    "The Dark Knight",
    "Titanic"
]

for movie in test_movies:

    print("\nRecommendations for:", movie)

    result = recommend(movie, top_n=5)

    if result.empty:
        print("Movie not found in dataset.")
    else:
        print(result)


# --------------------------------------------------
# TASK 6: STREAMLIT USER INTERFACE
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie to discover other movies "
    "with similar descriptions."
)

st.divider()

# Dropdown to select a movie
movie_list = sorted(df["title"].unique().tolist())

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

# Number of recommendations
top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# Generate recommendations
if st.button("Get Recommendations"):

    results = recommend(selected_movie, top_n)

    st.subheader("Recommended Movies")

    if results.empty:

        st.warning("No recommendations found.")

    else:

        for index, row in results.iterrows():

            st.write(
                f"**{index + 1}. {row['title']}**"
            )

            st.caption(
                f"Similarity Score: "
                f"{row['similarity_score']}"
            )
