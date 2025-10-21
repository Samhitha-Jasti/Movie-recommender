document.addEventListener('DOMContentLoaded', function() {
    const searchBtn = document.getElementById('searchBtn');
    const movieInput = document.getElementById('movieInput');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');

    // Autocomplete: create and attach datalist
    const datalist = document.createElement('datalist');
    datalist.id = 'movie-titles';
    movieInput.setAttribute('list', 'movie-titles');
    document.body.appendChild(datalist);

    let movieTitles = [];

    fetch('/api/movies')
        .then(res => res.json())
        .then(data => {
            movieTitles = data.movies || [];
            datalist.innerHTML = movieTitles.map(title => `<option value="${title}">`).join('');
        });

    // Recommend on Enter key
    movieInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') searchBtn.click();
    });

    // Main search and render function
    function fetchAndRender(movieTitle) {
        errorDiv.textContent = '';
        resultsDiv.innerHTML = '';
        if (!movieTitle) {
            errorDiv.textContent = 'Please enter a movie name!';
            return;
        }
        fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie_title: movieTitle })
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success || !data.recommendations || !data.recommendations.length) {
                errorDiv.textContent = data.error || "No recommendations found for this movie.";
                resultsDiv.innerHTML = "";
                return;
            } else {
                errorDiv.textContent = '';
            }
            resultsDiv.innerHTML = `<h2>Recommendations for "${data.query_movie}"</h2>`;
            data.recommendations.forEach(rec => {
                const card = document.createElement('div');
                card.className = 'rec-card';
                card.style = "background:#333;border-radius:8px;padding:12px;margin:12px 0;cursor:pointer;transition:background 0.2s;";
                card.title = `Find recommendations for '${rec.title}'`;
                card.innerHTML = `
                    <strong>${rec.title}</strong> <br>
                    Similarity: ${(rec.similarity*100).toFixed(1)}%<br>
                    Director: ${rec.director}<br>
                    Genres: ${rec.genres}<br>
                    Year: ${rec.year}<br>
                    IMDB: ${rec.imdb_rating}
                `;
                card.onclick = function() {
                    movieInput.value = rec.title;
                    fetchAndRender(rec.title);
                };
                resultsDiv.appendChild(card);
            });
        })
        .catch(err => {
            errorDiv.textContent = 'Server error: ' + err;
            resultsDiv.innerHTML = '';
        });
    }

    window.fetchAndRender = fetchAndRender;
    searchBtn.addEventListener('click', function() {
        fetchAndRender(movieInput.value.trim());
    });

});
