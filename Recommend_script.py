# %%
!pip list

# %%
!pip list --format=freeze > requirements.txt

# %%
import pandas as pd
df=pd.read_csv('movie_metadata.csv')
df

# %%
import numpy
numpy.version.version

# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df_backup=df.copy()
df_backup

# %%
print(df.isnull().sum())

x='''
color - not necessary
director_name - necessary, this data can be collected by online too. to recommend similar movies.
num_critic_for_reviews - don't think filling using mean, mode will be a good idea. to recommend movies on similar rating.
duration - important, can be filled by mean or mode?
director_facebook_likes - can't be filled randomly but maybe valuable data to determine the hype of movie. if good hype- can be recommended. or fill 0 for missing values as 0 also represents no likes.
actor_3_facebook_likes, actor_2_name  and actor_1_facebook_likes  - can be removed i guess as its less number but if they all belong to different rows, i am losing lot of data so i can just drop this column if necessary or fill 0 for missing values as 0 also represents no likes.
gross - really necessary to determine how good the movie is, as too many rows are missing, maybe just keep 0 instead of filling?
actor_1_name - can be ignored maybe as its categorical and you don't wanna be wrong?
actor_3_name - not necessary so can be dropped or is ignoring better?
facenumber_in_poster - is it even necessary? i guess so i believe depends on director?
plot_keywords - too hard to guess, so is there a way to ignore instead of filling?
num_user_for_reviews - can leave 0 for missing ones?
language - really necessary so how do i fill data based on other values of this column?
country - can be determine by language, director and actor name
content_rating - too many missing, can be left 0 instead of dropping?
budget - is necessary, as viewers may like high/low budget films. Can be kept 0 for unnecessary ones.
title_year - can be ignored and keep a unnecessary date?
actor_2_facebook_likes - can be filled with 0?
aspect_ratio - will be necessary to understand user preference, can be determined by movies made by same director? what if this is the only movie made by this director?
'''

# %% [markdown]
# 

# %%
df[df.isnull().any(axis=1)]

# %%
#df['color'].value_counts()
df['color'].value_counts(normalize=True)

# %%
df['color']=df['color'].fillna('Unknown')
df['color'].isnull().sum()

# %%
print(df['director_name'].value_counts())
#too many directors so no use of printing all

# %%
director_miss_perc=(df['director_name'].isnull().sum())*100/len(df)
print(director_miss_perc)

# %%
#missing director name is filled with 'unknown'
df['director_name']=df['director_name'].fillna('Unknown')
df['director_name'].isnull().sum()

# %%
df

# %%
#num_critic_for_reviews - not needed for model, column can be dropped
df.drop(columns=["num_critic_for_reviews"], inplace=True)
df

# %%
#duration may seem not important but helps differentiate short and long films.
#number of missing values are small, check if values are skewed using histogram
import matplotlib.pyplot as plt
print(df['duration'].skew())
duration_column=df['duration']
plt.hist(duration_column, bins=45)
plt.show()

# %%
#as duration column in right skewed, let's fill the missing values with median
df['duration'].fillna(df['duration'].median(), inplace=True)

# %%
print(df['duration'].isnull().sum())

# %%
df['actor_2_name']=df['actor_2_name'].fillna('Unknown')

# %%
df['actor_1_name']=df['actor_1_name'].fillna('Unknown')

# %%
df['actor_3_name']=df['actor_3_name'].fillna('Unknown')

# %%
df['language']=df['language'].fillna('Unknown')

# %%
df['country']=df['country'].fillna('Unknown')

# %%
print(df.columns)

# %%
#These columns will be usefull in ranking recommendations
#will dropp likes and lot_keywords data for now, it iwll be useful in future though so then you can remove step by step
df.drop(columns=["director_facebook_likes","actor_3_facebook_likes","actor_1_facebook_likes","gross","num_voted_users","cast_total_facebook_likes","facenumber_in_poster","plot_keywords","num_user_for_reviews","actor_2_facebook_likes","movie_facebook_likes","content_rating","budget","title_year","actor_2_facebook_likes","imdb_score", "aspect_ratio", "movie_imdb_link"], inplace=True)
df

# %%
print(df.columns)

# %%
df.info()

# %%
df_model1=df.copy()
df_model1

# %% [markdown]
# # Encoding

# %% [markdown]
# Column 'color' has 0.95 and 0.04 ratio of color and B/W movies so data is too uniform so encoding might not make much difference but let's keep the column because we dont want recommender to suggest B/W movies

# %% [markdown]
# 'Director name' - should be encoded, can keep the top 5-10 directors in different categories and rest all in last category.

# %%
print(df_model1['director_name'].value_counts())
#too many directors so no use of printing all

# %%
# Count frequency of each director
director_counts = df_model1['director_name'].value_counts()
# Filter directors that occur more than 5 times
directors_more_than_5 = director_counts[director_counts >= 5]
# Get the number of such directors
num_directors = len(directors_more_than_5)
print(f"Number of directors with more than 5 movies: {num_directors}")

# %%
print(df_model1['director_name'].nunique())

# %% [markdown]
# Director name column has 2398 unique values, so its better to do Binary encoding.

# %%
import sys
print(sys.executable)


# %%
conda env list

# %%
#!pip install category_encoders
#installed
import category_encoders as ce

# %%
df_model1

# %%
import category_encoders as ce

# List of categorical columns to encode
cat_columns = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']

for col in cat_columns:
    # 1. Count frequency of each value
    value_counts = df_model1[col].value_counts()
    
    # 2. Keep values that appear >= 2 times
    frequent_values = value_counts[value_counts >= 2].index
    
    # 3. Replace infrequent values with 'Other'
    df_model1[col] = df_model1[col].apply(lambda x: x if x in frequent_values else 'Other')
    
    # 4. Apply Binary Encoding
    encoder = ce.BinaryEncoder(cols=[col])
    encoded_df = encoder.fit_transform(df_model1[[col]])
    
    # 5. Drop original column and concat encoded columns
    df_model1 = df_model1.drop(col, axis=1)
    df_model1 = pd.concat([df_model1, encoded_df], axis=1)

# %%
print(df_model1.columns)

# %%
df_model1

# %%
print(df_model1['genres'].value_counts())

# %%
print(df_model1['genres'].nunique())

# %%
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Split the genre string into a list of genres
df_model1['genres_list'] = df_model1['genres'].apply(lambda x: x.split('|'))
# Step 2: Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Step 3: Apply the binarizer on the list of genres
genre_encoded = mlb.fit_transform(df_model1['genres_list'])
# Step 4: Convert the result into a DataFrame with genre names as column headers
df_genre = pd.DataFrame(genre_encoded, columns=mlb.classes_)

df_genre

# %%
df_model1 = pd.concat([df_model1, df_genre], axis=1)
df_model1.drop(['genres', 'genres_list'], axis=1, inplace=True)
df_model1

# %%
print(df_model1['language'].value_counts().to_string())

# %%
print(df_model1['language'].nunique())

# %%
language_counts = df_model1['language'].value_counts()
common_languages = language_counts[language_counts >= 3].index
df_model1['language'] =df_model1['language'].apply(lambda x: x if x in common_languages else 'Other')

# %%
language_encoded = pd.get_dummies(df_model1['language'], prefix='lang', dtype=int)
df_model1 = pd.concat([df_model1, language_encoded], axis=1)
df_model1.drop('language', axis=1, inplace=True)
df_model1

# %%
print(df_model1['country'].value_counts().to_string())

# %%
print(df_model1['country'].nunique())

# %%
country_counts = df_model1['country'].value_counts()
common_countries = country_counts[country_counts >= 3].index
df_model1['country'] =df_model1['country'].apply(lambda x: x if x in common_countries else 'Other')
df_model1

# %%
country_encoded = pd.get_dummies(df_model1['country'], prefix='country',dtype=int)
df_model1 = pd.concat([df_model1, country_encoded], axis=1)
df_model1.drop('country', axis=1, inplace=True)
df_model1

# %%
print(df_model1.columns.tolist())

# %%
color_dummies = pd.get_dummies(df_model1['color'], prefix='color').astype(int)
df_model1 = pd.concat([df_model1, color_dummies], axis=1)
df_model1.drop('color', axis=1, inplace=True)

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Scale duration between 0–1
df_model1['duration'] = scaler.fit_transform(df_model1[['duration']])


# %%
df_model1_backup=df_model1.copy()

# %%
df_model1

# %%
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Clean titles: remove leading/trailing spaces & non-breaking spaces (\xa0)
df_model1['movie_title'] = df_model1['movie_title'].str.replace('\xa0', '', regex=False).str.strip()

# 1️⃣ Separate movie titles (identifiers) and create feature matrix
movie_titles = df_model1['movie_title']                 # Keep titles for lookup
df_features = df_model1.drop('movie_title', axis=1)     # Only numeric features

# 2️⃣ Compute cosine similarity between movies
cosine_sim = cosine_similarity(df_features)
def recommend_movies(title, n=5):
    """
    Recommend top n movies similar to the given title.
    Prefers exact title match first, then partial matches.
    """
    # ✅ Exact match (ignores case)
    exact_matches = movie_titles[movie_titles.str.lower() == title.lower()]
    
    if not exact_matches.empty:
        idx = exact_matches.index[0]
        matched_title = exact_matches.iloc[0]
    else:
        # 🔍 Partial match fallback
        partial_matches = movie_titles[movie_titles.str.contains(title, case=False, na=False)]
        if partial_matches.empty:
            return f"No movie found matching '{title}'."
        idx = partial_matches.index[0]
        matched_title = partial_matches.iloc[0]

    # ✅ Get similarity scores for the found movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    movie_indices = [i[0] for i in sim_scores]
    return {
        "searched_for": matched_title,
        "recommendations": movie_titles.iloc[movie_indices].tolist()
    }


# %%
recommend_movies("The Dark Knight")

# %%
print(movie_titles)

# %%


# %%
df2=df_backup.copy()
print(df2.columns)

# %%

# CORRECTED COMPLETE A-Z HYBRID MOVIE RECOMMENDATION SYSTEM
# =========================================================
# This code is corrected for your ACTUAL dataset columns
# Combines TF-IDF + Categorical + Numerical features

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# STEP 1: DATA PREPARATION WITH PROPER NULL HANDLING
# ================================================================

def prepare_hybrid_data(df):
    print("🔄 Step 1: Preparing hybrid data with proper null handling...")

    # Create a copy to avoid modifying original
    df_hybrid = df.copy()

    # ============================================
    # TEXT FEATURES: Fill with EMPTY STRING
    # ============================================
    # CORRECTED: Using your actual text columns
    text_columns = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords']

    print(f"   📝 Processing text features: {text_columns}")
    for col in text_columns:
        if col in df_hybrid.columns:
            # Fill with empty string (BETTER than 'Unknown' for TF-IDF)
            df_hybrid[col] = df_hybrid[col].fillna('')
            print(f"      ✓ {col}: {df_hybrid[col].isnull().sum()} nulls remaining")

    # ============================================
    # CATEGORICAL FEATURES: Fill with 'Unknown'
    # ============================================
    # CORRECTED: Using your actual categorical columns
    categorical_columns = ['genres', 'language', 'country', 'content_rating', 'color']

    print(f"   🏷️  Processing categorical features: {categorical_columns}")
    for col in categorical_columns:
        if col in df_hybrid.columns:
            # Keep your existing approach: fill with 'Unknown'
            df_hybrid[col] = df_hybrid[col].fillna('Unknown')
            print(f"      ✓ {col}: {df_hybrid[col].isnull().sum()} nulls remaining")

    # ============================================
    # NUMERICAL FEATURES: Fill with median/mean
    # ============================================
    # CORRECTED: Using your actual numerical columns
    numerical_columns = ['duration', 'budget', 'title_year', 'imdb_score', 'gross', 'aspect_ratio']

    print(f"   🔢 Processing numerical features: {numerical_columns}")
    for col in numerical_columns:
        if col in df_hybrid.columns:
            # Fill with median (robust to outliers)
            median_val = df_hybrid[col].median()
            df_hybrid[col] = df_hybrid[col].fillna(median_val)
            print(f"      ✓ {col}: filled with {median_val}, {df_hybrid[col].isnull().sum()} nulls remaining")

    print(f"   ✅ Data preparation complete. Shape: {df_hybrid.shape}")
    return df_hybrid

# ================================================================
# STEP 2: TF-IDF FEATURE EXTRACTION (60% WEIGHT)
# ================================================================

def create_tfidf_features(df):
    """
    Create TF-IDF features from textual content
    CORRECTED: Using your actual dataset columns
    """
    print("🔄 Step 2: Creating TF-IDF features from text content...")

    # CORRECTED: Combine YOUR actual textual features into single text field
    # Give more weight to important fields by repeating them
    df['combined_text'] = (
        df['plot_keywords'].fillna('').str.replace('|', ' ') + ' ' +  # Plot keywords
        df['director_name'].fillna('') + ' ' + df['director_name'].fillna('') + ' ' +  # Double weight for director
        df['actor_1_name'].fillna('') + ' ' +  # Lead actor
        df['actor_2_name'].fillna('') + ' ' +  # Second actor  
        df['actor_3_name'].fillna('') + ' ' +  # Third actor
        df['genres'].fillna('').str.replace('|', ' ') + ' ' + df['genres'].fillna('').str.replace('|', ' ')  # Double weight for genres
    )

    # Clean the combined text
    df['combined_text'] = df['combined_text'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)

    # Apply TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,        # Reduced for efficiency in hybrid model
        stop_words='english',
        ngram_range=(1, 2),      # Unigrams and bigrams
        min_df=2,                # Ignore terms in < 2 documents
        max_df=0.8,              # Ignore terms in > 80% documents
        lowercase=True
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

    print(f"   ✅ TF-IDF matrix created: {tfidf_matrix.shape}")
    print(f"   📊 Sparsity: {(1 - tfidf_matrix.nnz / tfidf_matrix.size) * 100:.2f}%")

    return tfidf_matrix, tfidf_vectorizer

# ================================================================
# STEP 3: CATEGORICAL FEATURE EXTRACTION (30% WEIGHT)
# ================================================================

def create_categorical_features(df):
    """
    Create categorical features using your existing approach
    CORRECTED: Using your actual dataset columns
    """
    print("🔄 Step 3: Creating categorical features (keeping your existing work)...")

    categorical_matrices = []
    feature_names = []

    # ==========================================
    # A) GENRES: Multi-label binarization (your existing approach)
    # ==========================================
    if 'genres' in df.columns:
        print("   🎬 Processing genres with multi-label binarization...")

        # Split genres and create binary matrix
        df['genres_list'] = df['genres'].apply(lambda x: x.split('|') if x != 'Unknown' else [])
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(df['genres_list'])

        categorical_matrices.append(csr_matrix(genre_matrix))
        feature_names.extend([f'genre_{genre}' for genre in mlb.classes_])
        print(f"      ✓ Genres processed: {len(mlb.classes_)} unique genres")

    # ==========================================
    # B) LANGUAGE: One-hot encoding (your existing approach)
    # ==========================================
    if 'language' in df.columns:
        print("   🌍 Processing language features...")

        # Group rare languages as 'Other' (your existing approach)
        language_counts = df['language'].value_counts()
        common_languages = language_counts[language_counts >= 3].index
        df['language_grouped'] = df['language'].apply(
            lambda x: x if x in common_languages else 'Other'
        )

        # One-hot encode
        language_dummies = pd.get_dummies(df['language_grouped'], prefix='lang')
        categorical_matrices.append(csr_matrix(language_dummies.values))
        feature_names.extend([f'cat_{col}' for col in language_dummies.columns])
        print(f"      ✓ Languages processed: {len(common_languages)} common languages")

    # ==========================================
    # C) COUNTRY: One-hot encoding (your existing approach)
    # ==========================================
    if 'country' in df.columns:
        print("   🌎 Processing country features...")

        # Group rare countries as 'Other'
        country_counts = df['country'].value_counts()
        common_countries = country_counts[country_counts >= 3].index
        df['country_grouped'] = df['country'].apply(
            lambda x: x if x in common_countries else 'Other'
        )

        # One-hot encode
        country_dummies = pd.get_dummies(df['country_grouped'], prefix='country')
        categorical_matrices.append(csr_matrix(country_dummies.values))
        feature_names.extend([f'cat_{col}' for col in country_dummies.columns])
        print(f"      ✓ Countries processed: {len(common_countries)} common countries")

    # ==========================================
    # D) DIRECTOR/ACTOR ENCODING: Binary encoding (your existing approach)
    # ==========================================
    # CORRECTED: Using your actual actor column names
    actor_director_columns = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
    available_columns = [col for col in actor_director_columns if col in df.columns]

    if available_columns:
        print(f"   🎭 Processing director/actor features: {available_columns}")

        for col in available_columns:
            # Group infrequent values (your existing approach)
            value_counts = df[col].value_counts()
            frequent_values = value_counts[value_counts >= 2].index
            df[f'{col}_grouped'] = df[col].apply(
                lambda x: x if x in frequent_values else 'Other'
            )

            # Binary encoding (your existing approach)
            encoder = ce.BinaryEncoder(cols=[f'{col}_grouped'])
            encoded_df = encoder.fit_transform(df[[f'{col}_grouped']])

            categorical_matrices.append(csr_matrix(encoded_df.values))
            feature_names.extend([f'binary_{col}_{i}' for i in range(encoded_df.shape[1])])

        print(f"      ✓ Director/Actor encoding complete")

    # ==========================================
    # E) COLOR: One-hot encoding
    # ==========================================
    if 'color' in df.columns:
        color_dummies = pd.get_dummies(df['color'], prefix='color')
        categorical_matrices.append(csr_matrix(color_dummies.values))
        feature_names.extend([f'cat_{col}' for col in color_dummies.columns])
        print(f"      ✓ Color features processed")

    # ==========================================
    # F) CONTENT RATING: One-hot encoding
    # ==========================================
    if 'content_rating' in df.columns:
        rating_dummies = pd.get_dummies(df['content_rating'], prefix='rating')
        categorical_matrices.append(csr_matrix(rating_dummies.values))
        feature_names.extend([f'cat_{col}' for col in rating_dummies.columns])
        print(f"      ✓ Content rating features processed")

    # Combine all categorical features
    if categorical_matrices:
        categorical_combined = hstack(categorical_matrices)
        print(f"   ✅ Categorical features combined: {categorical_combined.shape}")
        return categorical_combined, feature_names
    else:
        print("   ⚠️  No categorical features found")
        return csr_matrix((len(df), 0)), []

# ================================================================
# STEP 4: NUMERICAL FEATURE EXTRACTION (10% WEIGHT)
# ================================================================

def create_numerical_features(df):
    """
    Create scaled numerical features
    CORRECTED: Using your actual numerical columns
    """
    print("🔄 Step 4: Creating numerical features...")

    # CORRECTED: Define numerical columns from your actual dataset
    numerical_columns = ['duration', 'budget', 'title_year', 'imdb_score', 'gross', 'aspect_ratio']
    available_numerical = [col for col in numerical_columns if col in df.columns]

    if available_numerical:
        print(f"   📊 Processing numerical columns: {available_numerical}")

        # Extract numerical data
        numerical_data = df[available_numerical].copy()

        # Scale features (your existing approach)
        scaler = MinMaxScaler()
        numerical_scaled = scaler.fit_transform(numerical_data)

        # Convert to sparse matrix for consistency
        numerical_matrix = csr_matrix(numerical_scaled)

        print(f"   ✅ Numerical features scaled: {numerical_matrix.shape}")
        return numerical_matrix, scaler, available_numerical
    else:
        print("   ⚠️  No numerical features found")
        return csr_matrix((len(df), 0)), None, []

# ================================================================
# STEP 5: HYBRID FEATURE COMBINATION WITH WEIGHTS
# ================================================================

def combine_hybrid_features(tfidf_matrix, categorical_matrix, numerical_matrix, 
                          tfidf_weight=0.5, categorical_weight=0.25, numerical_weight=0.25):
    """
    Combine all feature types with appropriate weights
    This is the KEY step that answers your question about mixing!
    """
    print("🔄 Step 5: Combining features with weights...")
    print(f"   ⚖️  Weights: TF-IDF={tfidf_weight}, Categorical={categorical_weight}, Numerical={numerical_weight}")

    # Apply weights to each feature type
    tfidf_weighted = tfidf_matrix * tfidf_weight
    categorical_weighted = categorical_matrix * categorical_weight  
    numerical_weighted = numerical_matrix * numerical_weight

    # Horizontally stack all weighted matrices
    combined_features = hstack([tfidf_weighted, categorical_weighted, numerical_weighted])

    print(f"   ✅ Combined feature matrix: {combined_features.shape}")
    print(f"   📊 Final breakdown:")
    print(f"      - TF-IDF features: {tfidf_matrix.shape[1]} × {tfidf_weight} = {tfidf_matrix.shape[1]} weighted")
    print(f"      - Categorical features: {categorical_matrix.shape[1]} × {categorical_weight} = {categorical_matrix.shape[1]} weighted")
    print(f"      - Numerical features: {numerical_matrix.shape[1]} × {numerical_weight} = {numerical_matrix.shape[1]} weighted")
    print(f"      - Total features: {combined_features.shape[1]}")

    return combined_features

# ================================================================
# STEP 6: SIMILARITY CALCULATION
# ================================================================

def calculate_hybrid_similarity(combined_features):
    """
    Calculate cosine similarity on combined feature matrix
    """
    print("🔄 Step 6: Calculating hybrid similarity matrix...")

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(combined_features)

    print(f"   ✅ Similarity matrix calculated: {similarity_matrix.shape}")
    return similarity_matrix

# ================================================================
# STEP 7: RECOMMENDATION FUNCTION
# ================================================================

def create_hybrid_recommender(df, similarity_matrix):
    """
    Create the final recommendation function
    Same logic as your existing function, but with hybrid similarity!
    """

    def recommend_movies_hybrid(title, n_recommendations=5, min_similarity=0.1):
        """
        Get hybrid recommendations combining TF-IDF + Categorical + Numerical features
        """
        try:
            # Clean movie titles (your existing approach)
            movie_titles = df['movie_title'].str.replace('\xa0', '').str.strip()

            # Find movie index (your existing logic)
            exact_matches = movie_titles[movie_titles.str.lower() == title.lower()]
            if not exact_matches.empty:
                idx = exact_matches.index[0]
                matched_title = exact_matches.iloc[0]
            else:
                partial_matches = movie_titles[movie_titles.str.contains(title, case=False, na=False)]
                if partial_matches.empty:
                    return f"❌ Movie '{title}' not found in database"
                idx = partial_matches.index[0]
                matched_title = partial_matches.iloc[0]

            # Get similarity scores
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude self

            # Filter by minimum similarity
            sim_scores = [score for score in sim_scores if score[1] >= min_similarity]

            # Get top recommendations
            top_recommendations = sim_scores[:n_recommendations]

            # Format recommendations
            recommendations = []
            for movie_idx, similarity_score in top_recommendations:
                movie_info = {
                    'title': movie_titles.iloc[movie_idx],
                    'similarity_score': round(similarity_score, 3),
                    'director': df.iloc[movie_idx].get('director_name', 'Unknown'),
                    'genres': df.iloc[movie_idx].get('genres', 'Unknown'),
                    'year': df.iloc[movie_idx].get('title_year', 'Unknown'),
                    'imdb_score': df.iloc[movie_idx].get('imdb_score', 'Unknown')
                }
                recommendations.append(movie_info)

            return {
                'searched_for': matched_title,
                'recommendations': recommendations,
                'total_found': len(recommendations),
                'feature_types_used': ['TF-IDF (60%)', 'Categorical (30%)', 'Numerical (10%)']
            }

        except Exception as e:
            return f"❌ Error processing recommendation: {str(e)}"

    return recommend_movies_hybrid

# ================================================================
# STEP 8: MAIN IMPLEMENTATION FUNCTION
# ================================================================

def implement_complete_hybrid_system(df_original):
    """
    MAIN FUNCTION: Complete A-Z implementation
    CORRECTED for your actual dataset columns

    Input: Your original dataframe (BEFORE dropping text columns)
    Output: Hybrid recommendation function
    """
    print("🚀 IMPLEMENTING COMPLETE HYBRID RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("✅ CORRECTED for your actual dataset columns!")

    # Step 1: Prepare data with proper null handling
    df_prepared = prepare_hybrid_data(df_original)

    # Step 2: Create TF-IDF features (60% weight)
    tfidf_matrix, tfidf_vectorizer = create_tfidf_features(df_prepared)

    # Step 3: Create categorical features (30% weight) 
    categorical_matrix, categorical_names = create_categorical_features(df_prepared)

    # Step 4: Create numerical features (10% weight)
    numerical_matrix, scaler, numerical_names = create_numerical_features(df_prepared)

    # Step 5: Combine features with weights
    combined_features = combine_hybrid_features(
        tfidf_matrix, categorical_matrix, numerical_matrix,
        tfidf_weight=0.6, categorical_weight=0.3, numerical_weight=0.1
    )

    # Step 6: Calculate similarity
    similarity_matrix = calculate_hybrid_similarity(combined_features)

    # Step 7: Create recommendation function
    recommend_movies = create_hybrid_recommender(df_prepared, similarity_matrix)

    print("\n🎉 HYBRID SYSTEM IMPLEMENTATION COMPLETE!")
    print("=" * 45)
    print("✅ TF-IDF features: plot_keywords + director + actors + genres")
    print("✅ Categorical features: genres + language + country + color + rating")  
    print("✅ Numerical features: duration + budget + year + score + gross + aspect_ratio")
    print("✅ Weighted combination: Optimal feature balance")
    print("✅ CORRECTED for your actual dataset columns!")

    return recommend_movies, {
        'similarity_matrix': similarity_matrix,
        'tfidf_vectorizer': tfidf_vectorizer,
        'scaler': scaler,
        'feature_breakdown': {
            'tfidf_shape': tfidf_matrix.shape,
            'categorical_shape': categorical_matrix.shape,
            'numerical_shape': numerical_matrix.shape,
            'combined_shape': combined_features.shape
        }
    }

# ================================================================
# STEP 9: TESTING FUNCTION
# ================================================================

def test_hybrid_system(recommend_function):
    """
    Test the hybrid recommendation system
    """
    print("\n🧪 TESTING HYBRID RECOMMENDATION SYSTEM")
    print("=" * 45)

    test_movies = ["The Dark Knight", "Avatar", "Inception", "Titanic"]

    for movie in test_movies:
        print(f"\n🎬 Testing recommendations for '{movie}':")
        print("-" * 50)

        result = recommend_function(movie, n_recommendations=5)

        if isinstance(result, dict) and 'recommendations' in result:
            print(f"✅ Found: {result['searched_for']}")
            print(f"🔧 Feature types: {', '.join(result['feature_types_used'])}")
            print("📋 Top recommendations:")

            for i, rec in enumerate(result['recommendations'], 1):
                print(f"   {i}. {rec['title']}")
                print(f"      📊 Similarity: {rec['similarity_score']}")
                print(f"      🎭 Director: {rec['director']}")
                print(f"      🎪 Genres: {rec['genres']}")
                print(f"      📅 Year: {rec['year']}")
                print(f"      ⭐ IMDB: {rec['imdb_score']}")
        else:
            print(result)

# ================================================================
# USAGE INSTRUCTIONS FOR YOUR ACTUAL DATASET
# ================================================================

print("CORRECTED A-Z IMPLEMENTATION FOR YOUR DATASET!")
print("=" * 50)

usage_instructions = """
🔧 HOW TO USE THIS CORRECTED CODE:

1. REPLACE your existing similarity calculation with:

   # Load your original data 
   df_original = pd.read_csv('movie_metadata.csv')

   # Implement corrected hybrid system
   recommend_movies_hybrid, system_info = implement_complete_hybrid_system(df_original)

   # Test the system
   test_hybrid_system(recommend_movies_hybrid)

   # Test specifically with The Dark Knight
   result = recommend_movies_hybrid("The Dark Knight", n_recommendations=5)
   print(result)

2. WHAT'S CORRECTED:
   ✅ Removed 'overview' (not in your dataset)
   ✅ Removed 'cast' (not in your dataset)  
   ✅ Uses 'actor_1_name', 'actor_2_name', 'actor_3_name'
   ✅ Uses 'plot_keywords' for text content
   ✅ Added 'gross', 'aspect_ratio' to numerical features
   ✅ All columns match your actual dataset

3. TEXT FEATURES USED:
   📝 director_name, actor_1_name, actor_2_name, actor_3_name, plot_keywords
   📝 These are combined and processed with TF-IDF

4. EXPECTED RESULTS:
   🎯 Much better recommendations for "The Dark Knight"
   🎯 Should find Batman movies, Christopher Nolan films
   🎯 Combines content understanding with structural patterns
"""

print(usage_instructions)

# %%
   # Load your original data 
   df_original = pd.read_csv('movie_metadata.csv')

   # Implement corrected hybrid system
   recommend_movies_hybrid, system_info = implement_complete_hybrid_system(df_original)

   # Test the system
   test_hybrid_system(recommend_movies_hybrid)

   # Test specifically with The Dark Knight
   result = recommend_movies_hybrid("The Dark Knight", n_recommendations=5)
   print(result)

# %%



