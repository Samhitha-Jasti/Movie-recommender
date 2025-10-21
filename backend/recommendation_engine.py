import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommendationEngine:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
        self.load_data()

    def load_data(self):
        """Load and prepare the movie dataset"""
        try:
            # Load your movie metadata
            self.movies_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'movie_metadata.csv'))
            # Clean up the movie title column
            self.movies_df['movie_title'] = self.movies_df['movie_title'].str.replace('\xa0', '', regex=False).str.strip().str.lower()
            self.build_similarity_matrix()
            print(f"Loaded {len(self.movies_df)} movies")
            print("Columns in movies_df:", list(self.movies_df.columns))
        except Exception as e:
            print(f"Error loading data: {e}")

    def build_similarity_matrix(self):
        # Combine relevant features for similarity calculation
        features = (
            self.movies_df['movie_title'].fillna('') + ' ' +
            self.movies_df['director_name'].fillna('') + ' ' +
            self.movies_df['genres'].fillna('') + ' ' +
            self.movies_df['actor_1_name'].fillna('') + ' ' +
            self.movies_df['actor_2_name'].fillna('') + ' ' +
            self.movies_df['actor_3_name'].fillna('') + ' ' +
            self.movies_df['plot_keywords'].fillna('')
        )
        # Create TF-IDF vectorizer and compute matrix
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(features)
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("Similarity matrix shape:", self.similarity_matrix.shape)

    def get_recommendations(self, movie_title, num_recommendations=5):
        print("Searching for:", movie_title)
        print("Sample from movie_title column:", self.movies_df['movie_title'].head(10).tolist())
        try:
            search_title = movie_title.replace('\xa0', '').strip().lower()
            matches = self.movies_df[self.movies_df['movie_title'] == search_title]
            if len(matches) == 0:
                raise ValueError(f"No movie found with title '{movie_title}'")
            movie_idx = matches.index[0]
            sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_movies = sim_scores[1:num_recommendations+1]
            recommendations = []
            for idx, score in top_movies:
                movie_data = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie_data['movie_title'],
                    'similarity': round(score, 3),
                    'director': movie_data.get('director_name', 'Unknown'),
                    'genres': movie_data.get('genres', 'Unknown'),
                    'year': movie_data.get('title_year', 'Unknown'),
                    'imdb_rating': movie_data.get('imdb_score', 'N/A')
                })
            print("Similarity matrix shape:", self.similarity_matrix.shape)
            print(recommendations)
            return {
                'success': True,
                'query_movie': movie_title,
                'recommendations': recommendations
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_all_movie_titles(self):
        """Return list of all available movie titles for autocomplete"""
        if self.movies_df is not None:
            return self.movies_df['movie_title'].tolist()
        return []

if __name__ == "__main__":
    engine = MovieRecommendationEngine()
    result = engine.get_recommendations("Inception")
    print(result)