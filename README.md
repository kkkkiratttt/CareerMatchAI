# CareerMatch AI

A standalone resume-first job discovery web app.

## Features

- Upload one PDF, DOCX, or TXT resume.
- Extract resume text, skills, high-signal terms, and possible target roles.
- Search the live web using Mwmbl queries derived from the resume.
- Check discovered job pages and scrape only where permitted by robots.txt; LinkedIn is skipped.
- Rank accessible job pages using local TF-IDF similarity and term coverage.
- Review search diagnostics, source URLs, and downloadable HTML/CSV/Markdown reports.

## AI/API dependencies

No Groq, Gemini, OpenRouter, OpenAI, or paid LLM API key is required. Matching is performed locally with scikit-learn TF-IDF. Live discovery uses the public Mwmbl search endpoint; coverage and page accessibility depend on that service and the destination websites.

## Local run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r CareerMatchAI_RenderApp/requirements.txt
cd CareerMatchAI_RenderApp
python app.py
```

Open `http://127.0.0.1:10000`.

## Render deployment

This repository includes `render.yaml`. Create a Render Blueprint for this repository's `main` branch. The build decodes and extracts the application source archive from `source_parts/`, installs Python dependencies, and starts Gunicorn on Render's `$PORT`.

- Service: `careermatch-ai`
- Runtime: Python
- Plan: Free
- Health check: `/healthz`
- Required API secrets: None

Render free services may spin down after inactivity. A first request after sleeping can take time to start.

## Notes

- Maximum resume size: 10 MB.
- Supported formats: PDF, DOCX, TXT.
- Uploaded resume bytes are processed in memory and are not stored as a profile.
- Only one search runs at a time and a per-IP cooldown helps protect the shared Mwmbl search quota.
- Job listings and snippets can be incomplete or stale. Verify application details on the original employer page.
