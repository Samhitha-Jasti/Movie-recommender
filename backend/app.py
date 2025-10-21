# backend/app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from recommendation_engine import MovieRecommendationEngine
import os

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')
CORS(app)  # Enable cross-origin requests

# Initialize the recommendation engine
recommender = MovieRecommendationEngine()

@app.route('/')
def home():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API endpoint for getting movie recommendations"""
    try:
        data = request.get_json()
        movie_title = data.get('movie_title', '').strip()
        
        if not movie_title:
            return jsonify({
                'success': False, 
                'error': 'Movie title is required'
            })
        
        # Get recommendations
        result = recommender.get_recommendations(movie_title)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """API endpoint to get all available movies for autocomplete"""
    try:
        movies = recommender.get_all_movie_titles()
        return jsonify({
            'success': True,
            'movies': movies
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'movies_loaded': len(recommender.movies_df) if recommender.movies_df is not None else 0
    })

if __name__ == '__main__':
    print("Starting Movie Recommendation API...")
    print("Visit http://localhost:5000 to use the web application")
    app.run(debug=True, host='0.0.0.0', port=5000)
