# CareerMatch AI — Flask Application Source

This directory contains the readable Python source used by the deployed CareerMatch AI web application.

## Files

- `app.py` — Flask app, HTTP routes, background job execution, progress tracking, upload validation, and report downloads.
- `core.py` — resume parsing, skill/role extraction, Mwmbl search, robots.txt-aware page retrieval, JobPosting metadata parsing, TF-IDF ranking, and report generation.
- `requirements.txt` — Python dependencies.
- `.python-version` — Python runtime version.

The HTML template and CSS/JavaScript frontend are included in the application source archive under the repository's `source_parts/` directory. Render reconstructs that archive during the build.

## Run locally

From this directory:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:10000`.

For production, the Render service runs Gunicorn and binds to the port provided by the `PORT` environment variable.

## Matching method

The current pipeline uses TF-IDF vectors, cosine similarity, and weighted term coverage. The final match index is:

```text
Match Index = 0.75 × Text Similarity + 0.25 × Term Coverage
```

This is a text-relevance score, not a probability of getting hired. The project does not require a paid generative-AI API key.
