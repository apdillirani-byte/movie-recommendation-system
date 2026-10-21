import streamlit as st
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CineMatch")
st.subheader("Discover your next favourite movie!")
st.write("Search a movie, explore genres, and discover similar movies.")

@st.cache_data
def load_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = Path("movies_dataset.zip")

    if not zip_path.exists():
        urllib.request.urlretrieve(url, str(zip_path))

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data")

    base = Path("data/ml-latest-small")
    movie_data = pd.read_csv(base / "movies.csv")
    rating_data = pd.read_csv(base / "ratings.csv")

    movie_data = movie_data.dropna(subset=["title"]).copy()
    movie_data = movie_data.drop_duplicates(subset=["title"])
    movie_data["genres"] = movie_data["genres"].fillna("")

    rating_summary = rating_data.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    movie_data = movie_data.merge(
        rating_summary, on="movieId", how="left"
    )
    movie_data["average_rating"] = movie_data["average_rating"].fillna(0)
    movie_data["rating_count"] = movie_data["rating_count"].fillna(0).astype(int)

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b"
    )
    matrix = vectorizer.fit_transform(
        movie_data["genres"].str.replace("|", " ", regex=False)
    )

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(matrix)

    return movie_data.reset_index(drop=True), matrix, model


try:
    movies, genre_matrix, model = load_data()
except Exception as e:
    st.error("Could not load the dataset. Please refresh and try again.")
    st.stop()

st.success(f"Loaded {len(movies):,} movies")

all_genres = sorted({
    genre
    for value in movies["genres"]
    for genre in value.split("|")
    if genre and genre != "(no genres listed)"
})

selected_genre = st.selectbox(
    "🎭 Filter by genre",
    ["All genres"] + all_genres
)

if selected_genre == "All genres":
    available_movies = movies
else:
    available_movies = movies[
        movies["genres"].str.split("|").apply(
            lambda genres: selected_genre in genres
        )
    ]

movie_titles = sorted(available_movies["title"].tolist())

if not movie_titles:
    st.warning("No movies found for this genre.")
    st.stop()

selected_title = st.selectbox(
    "🔍 Choose a movie",
    movie_titles
)

number_to_show = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("✨ Recommend Similar Movies", type="primary"):
    selected_index = movies.index[
        movies["title"] == selected_title
    ][0]

    distances, indices = model.kneighbors(
        genre_matrix[selected_index],
        n_neighbors=min(number_to_show + 1, len(movies))
    )

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == selected_index:
            continue

        row = movies.iloc[idx]
        results.append({
            "Movie": row["title"],
            "Genres": row["genres"].replace("|", ", "),
            "⭐ Rating": round(row["average_rating"], 1),
            "Rating Count": int(row["rating_count"]),
            "Genre Similarity": f"{max(0, 1-distance)*100:.1f}%"
        })

        if len(results) >= number_to_show:
            break

    chosen = movies.iloc[selected_index]
    st.markdown("---")
    st.subheader("🎯 Your Selected Movie")
    st.write(f"**{chosen['title']}**")
    st.write(f"Genres: {chosen['genres'].replace('|', ', ')}")
    st.write(f"⭐ Average user rating: {chosen['average_rating']:.1f}/5")
    st.caption(f"Based on {chosen['rating_count']} user ratings")

    st.subheader("🍿 Recommended Movies")
    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")
st.caption(
    "MovieLens dataset | Recommendations use genre similarity. "
    "Ratings are averages of user ratings in the dataset."
)
