"""
Movie Recommendation Engine
Content-based filtering with TF-IDF and cosine similarity (TMDB 5000 Movies).
"""

import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
)

DEFAULT_MOVIE = "Avatar"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading model...")
def load_data():
    movies = pd.read_csv("movies_clean.csv")
    tfidf_matrix = sp.load_npz("tfidf_matrix.npz")
    return movies, tfidf_matrix


try:
    movies, tfidf_matrix = load_data()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `movies_clean.csv` and "
        "`tfidf_matrix.npz` are in the same folder as `app.py`."
    )
    st.stop()


# --------------------------------------------------------------------------
# Recommendation logic
# --------------------------------------------------------------------------
def get_recommendations(title: str, n: int, min_votes: int) -> pd.DataFrame:
    """Return the n movies most similar to `title`."""
    idx = movies.index[movies["title"] == title][0]

    # Similarity of the selected movie against every movie
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    scores[idx] = -1                                        # exclude the movie itself
    scores[movies["vote_count"].values < min_votes] = -1    # exclude low-vote movies

    top = scores.argsort()[::-1][:n]
    result = movies.loc[top, ["title", "genres", "director", "vote_average"]].copy()
    result["similarity"] = scores[top].round(3)
    result.columns = ["Title", "Genres", "Director", "Rating", "Similarity"]
    result = result.reset_index(drop=True)
    result.index = result.index + 1
    return result


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    n = st.slider("Number of recommendations", 5, 20, 10)
    min_votes = st.slider(
        "Minimum vote count",
        0, 500, 20,
        help="Hides movies with very few votes, which have little data to compare.",
    )

    st.divider()
    st.caption(
        f"Dataset: TMDB 5000 Movies  \n"
        f"{len(movies):,} movies indexed"
    )

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.title("🎬 Movie Recommendation Engine")
st.write("Choose a movie and discover similar ones.")

titles = sorted(movies["title"].tolist())
default_index = titles.index(DEFAULT_MOVIE) if DEFAULT_MOVIE in titles else 0
selected = st.selectbox("Select a movie (type to search)", titles, index=default_index)

# Details of the selected movie
info = movies[movies["title"] == selected].iloc[0]
col1, col2, col3 = st.columns(3)
col1.metric("Director", info["director"] if pd.notna(info["director"]) else "Unknown")
col2.metric("Rating", f"{info['vote_average']:.1f} / 10")
col3.metric("Votes", f"{int(info['vote_count']):,}")
st.caption(f"Genres: {info['genres']}")

st.divider()

# Recommendations (updates automatically when the selection or settings change)
result = get_recommendations(selected, n, min_votes)

st.subheader(f"Movies similar to {selected}")

if result.empty:
    st.warning("No movies match the current settings. Try lowering the minimum vote count.")
else:
    st.dataframe(
        result,
        width="stretch",
        column_config={
            "Rating": st.column_config.NumberColumn(format="%.1f"),
            "Similarity": st.column_config.ProgressColumn(
                format="%.3f",
                min_value=0.0,
                max_value=float(max(result["Similarity"].max(), 0.01)),
                help="Cosine similarity. Bars are relative to the top match.",
            ),
        },
    )

    st.download_button(
        "Download results (CSV)",
        data=result.to_csv(index_label="Rank").encode("utf-8"),
        file_name=f"similar_to_{selected.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# About
# --------------------------------------------------------------------------
with st.expander("How does it work?"):
    st.markdown(
        """
        1. Genres, keywords, plot overview, cast and director are combined into one text field per movie.
        2. The text is converted to numeric vectors with **TF-IDF**, using single words and word pairs.
        3. **Cosine similarity** ranks all movies by how close they are to the selected movie.
        4. The movie itself and movies below the minimum vote count are removed from the list.

        The director is weighted more heavily than other fields, so films by the same director
        tend to rank near the top.
        """
    )