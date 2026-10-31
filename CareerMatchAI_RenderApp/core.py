"""CareerMatch AI pipeline adapted from the user's uploaded Colab notebook."""
import io
import zipfile
import shutil
import os
import re
import json
import html
import time
import tempfile
import warnings
import urllib.robotparser
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
try:
    import trafilatura
except ImportError:
    trafilatura = None
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

SEARCH_API_URL = "https://api.mwmbl.org/api/v2/search/"
SEARCH_DELAY_SECONDS = 1.2
MAX_RESULTS_PER_QUERY = 10
MAX_PAGES_TO_SCRAPE = 20
MAX_PAGE_CHARS = 22000
PAGE_DELAY_SECONDS = 1.5
SCRAPER_USER_AGENT = "CareerMatchAIStudentProject/1.0"
BLOCKED_DOMAINS = {"linkedin.com"}

STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "resume", "curriculum", "vitae", "education", "experience", "projects",
    "project", "skills", "responsibilities", "university", "college", "built",
    "developed", "worked", "using", "responsible", "strong", "good", "knowledge",
    "understanding", "ability", "various", "different", "including", "candidate",
    "team", "work", "student", "summary", "profile", "objective", "contact"
})

SKILLS_HEADING = re.compile(
    r"^\s*(?:technical\s+)?(?:key\s+)?(?:skills|skill\s+set|technologies|tools|technical\s+stack|programming\s+languages|core\s+competencies)\s*[:\-]?\s*$",
    re.I
)
SECTION_HEADING = re.compile(
    r"^\s*(?:education|academic|work experience|employment|professional experience|projects|certifications|awards|summary|profile|objective|achievements|experience\s*&\s*activities)\s*[:\-]?\s*$",
    re.I
)

def extract_resume_text(filename, file_bytes):
    extension = os.path.splitext(filename.lower())[1]
    if extension == ".pdf":
        pdf = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                parts.append(page.extract_text() or "")
            except Exception as exc:
                print(f"Warning: could not read PDF page {page_number}: {exc}")
        text = "\n".join(parts)
    elif extension == ".docx":
        doc = Document(io.BytesIO(file_bytes))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        text = "\n".join(parts)
    elif extension == ".txt":
        text = file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported file format. Upload a PDF, DOCX, or TXT resume.")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def redact_contact_details(text):
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", " ", text)
    return text

def extract_skill_section(text):
    in_skills = False
    found, seen = [], set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if SKILLS_HEADING.match(line):
            in_skills = True
            continue
        if in_skills and SECTION_HEADING.match(line):
            in_skills = False
            continue
        if not in_skills:
            continue
        line = re.sub(r"^[A-Za-z][A-Za-z0-9 /&+.#-]{1,35}:\s*", "", line)
        for part in re.split(r"[,;|•·]+", line):
            phrase = re.sub(r"^[\s:•·\-–]+|[\s.]+$", "", part).strip()
            phrase = re.sub(r"\s+", " ", phrase)
            if not phrase or len(phrase) > 55 or len(phrase.split()) > 6:
                continue
            if phrase.casefold() in STOP_WORDS or re.search(r"@|https?://|\b\d{6,}\b", phrase):
                continue
            if phrase.casefold() not in seen:
                found.append(phrase)
                seen.add(phrase.casefold())
    return found

def extract_fallback_terms(text, top_n=25):
    clean = redact_contact_details(text)
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words=list(STOP_WORDS),
        ngram_range=(1, 2), max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]{1,}\b"
    )
    try:
        matrix = vectorizer.fit_transform([clean])
    except ValueError:
        return []
    terms = vectorizer.get_feature_names_out()
    weights = matrix.toarray().ravel()
    order = np.argsort(weights)[::-1]
    return [terms[i] for i in order if weights[i] > 0][:top_n]

def extract_target_roles(text):
    pattern = r"\b(?:seeking|looking for|interested in|targeting)\s+(?:an?\s+)?(.*?)(?=\b(?:opportunity|position|role|roles)\b|[.\n])"
    matches = re.findall(pattern, text, flags=re.I | re.S)
    roles, seen = [], set()
    for match in matches:
        match = re.sub(r"\s+", " ", match).strip(" ,;:-")
        for part in re.split(r",|\bor\b|\band\b", match, flags=re.I):
            title = re.sub(r"^(?:an?|the)\s+", "", part.strip(), flags=re.I)
            if 2 <= len(title.split()) <= 7 and len(title) <= 65:
                if title.casefold() not in seen:
                    roles.append(title)
                    seen.add(title.casefold())
    return roles[:6]

def build_resume_signals(resume_text):
    resume_skills = extract_skill_section(resume_text)
    fallback_terms = extract_fallback_terms(resume_text)
    target_roles = extract_target_roles(resume_text)
    resume_terms, seen = [], set()
    for term in resume_skills + fallback_terms:
        if term.casefold() not in seen:
            resume_terms.append(term)
            seen.add(term.casefold())
    if not resume_terms and not target_roles:
        raise ValueError("Could not extract useful search terms from this resume. Add a clear Skills section or more descriptive experience/project text.")
    return resume_skills, fallback_terms, target_roles, resume_terms

def make_search_queries(roles, terms, max_queries=5):
    candidates = []
    for role in roles:
        candidates.append(f'"{role}" jobs')
    clean_terms = [x for x in terms if len(x.split()) <= 4]
    for i in range(0, min(len(clean_terms), 10), 2):
        pair = clean_terms[i:i + 2]
        if pair:
            candidates.append(" ".join(pair) + " jobs vacancies")
    candidates.extend([f'"{term}" remote jobs' for term in clean_terms[:3]])
    queries, seen = [], set()
    for q in candidates:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.casefold() not in seen:
            queries.append(q)
            seen.add(q.casefold())
        if len(queries) >= max_queries:
            break
    return queries
def make_search_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "CareerMatchAI-StudentProject/1.0 (contact: educational project)",
        "Accept": "application/json"
    })
    return session

def mwmbl_search(query, session, diagnostics):
    record = {"query": query, "http_status": None, "results": 0, "error": ""}
    try:
        response = session.get(SEARCH_API_URL, params={"q": query}, timeout=30)
        record["http_status"] = response.status_code
        if response.status_code == 429:
            record["error"] = "Monthly quota or request-rate limit reached (HTTP 429)"
            diagnostics.append(record)
            return None
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            record["error"] = "Unexpected response format; expected a results list"
            diagnostics.append(record)
            return None
        results = payload["results"]
        record["results"] = len(results)
        record["number_of_results"] = payload.get("number_of_results")
        record["monthly_usage"] = payload.get("monthly_usage")
        record["monthly_limit"] = payload.get("monthly_limit")
        diagnostics.append(record)
        return results
    except requests.exceptions.Timeout:
        record["error"] = "Request timed out"
    except requests.exceptions.ConnectionError as exc:
        record["error"] = f"Connection error: {str(exc)[:160]}"
    except requests.exceptions.HTTPError as exc:
        record["error"] = f"HTTP error: {str(exc)[:160]}"
    except requests.exceptions.RequestException as exc:
        record["error"] = f"Request error: {str(exc)[:160]}"
    except ValueError as exc:
        record["error"] = f"Invalid JSON: {str(exc)[:160]}"
    diagnostics.append(record)
    return None

def get_domain(url):
    return (urlparse(url).hostname or "").lower().lstrip("www.")

def domain_is_blocked(domain):
    return any(domain == blocked or domain.endswith("." + blocked) for blocked in BLOCKED_DOMAINS)

def can_fetch_url(url, session, robots_cache):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False, "Invalid URL"
    domain = get_domain(url)
    if domain_is_blocked(domain):
        return False, "Blocked by project policy (site terms prohibit scraping)"
    root = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = root + "/robots.txt"
    if robots_url not in robots_cache:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = session.get(robots_url, timeout=15)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                robots_cache[robots_url] = parser
            elif response.status_code == 404:
                robots_cache[robots_url] = None
            else:
                robots_cache[robots_url] = "UNKNOWN_BLOCK"
        except requests.RequestException:
            robots_cache[robots_url] = "UNKNOWN_BLOCK"
    parser = robots_cache[robots_url]
    if parser == "UNKNOWN_BLOCK":
        return False, "Could not verify robots.txt; skipped conservatively"
    if parser is not None and not parser.can_fetch(SCRAPER_USER_AGENT, url):
        return False, "Disallowed by robots.txt"
    return True, "Allowed by robots.txt or no robots file"

def find_jobposting_jsonld(soup):
    def walk(obj):
        found = []
        if isinstance(obj, dict):
            obj_type = obj.get("@type")
            if obj_type == "JobPosting" or (isinstance(obj_type, list) and "JobPosting" in obj_type):
                found.append(obj)
            for value in obj.values():
                found.extend(walk(value))
        elif isinstance(obj, list):
            for value in obj:
                found.extend(walk(value))
        return found
    postings = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw.strip())
            postings.extend(walk(data))
        except (json.JSONDecodeError, TypeError):
            continue
    return postings

def jsonld_org_name(value):
    if isinstance(value, dict):
        return value.get("name") or ""
    if isinstance(value, list):
        return ", ".join(filter(None, [jsonld_org_name(x) for x in value]))
    return str(value or "")

def jsonld_location(value):
    if isinstance(value, dict):
        address = value.get("address", {})
        if isinstance(address, dict):
            parts = [address.get("addressLocality"), address.get("addressRegion"), address.get("addressCountry")]
            parts = [str(x) for x in parts if x]
            if parts:
                return ", ".join(parts)
        return value.get("name") or ""
    if isinstance(value, list):
        return "; ".join(filter(None, [jsonld_location(x) for x in value]))
    return str(value or "")

def scrape_job_page(search_item, session, robots_cache, last_request_by_domain):
    url = search_item["url"]
    domain = get_domain(url)
    allowed, reason = can_fetch_url(url, session, robots_cache)
    base = {
        **search_item,
        "company": "", "job_title": search_item["title"], "job_location": "",
        "date_posted": "", "employment_type": "",
        "description": search_item["snippet"], "page_text": search_item["snippet"],
        "scrape_status": reason, "structured_jobposting": False
    }
    if not allowed:
        return base
    elapsed = time.time() - last_request_by_domain[domain]
    if elapsed < PAGE_DELAY_SECONDS:
        time.sleep(PAGE_DELAY_SECONDS - elapsed)
    try:
        response = session.get(
            url, headers={"User-Agent": SCRAPER_USER_AGENT}, timeout=25, allow_redirects=True
        )
        last_request_by_domain[domain] = time.time()
        if response.status_code in {401, 403, 429}:
            base["scrape_status"] = f"HTTP {response.status_code}; source restricted scraping"
            return base
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and "xhtml" not in content_type:
            base["scrape_status"] = f"Skipped non-HTML response ({content_type or 'unknown type'})"
            return base
        soup = BeautifulSoup(response.text, "html.parser")
        postings = find_jobposting_jsonld(soup)
        posting = postings[0] if postings else None
        title = company = location = date_posted = employment_type = description = ""
        if posting:
            title = str(posting.get("title") or "")
            company = jsonld_org_name(posting.get("hiringOrganization"))
            location = jsonld_location(posting.get("jobLocation"))
            if posting.get("jobLocationType") == "TELECOMMUTE":
                location = (location + " | Remote").strip(" |")
            date_posted = str(posting.get("datePosted") or "")
            employment_type = str(posting.get("employmentType") or "")
            description = BeautifulSoup(str(posting.get("description") or ""), "html.parser").get_text(" ", strip=True)
        page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if not title:
            title_tag = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
            title = title_tag.get("content", "") if title_tag else page_title
        site_name = soup.find("meta", property="og:site_name")
        if not company and site_name:
            company = site_name.get("content", "")
        if not description:
            try:
                if trafilatura is not None:
                    description = trafilatura.extract(response.text, include_comments=False, include_tables=True, favor_precision=True) or ""
                else:
                    description = soup.get_text(" ", strip=True)
            except Exception:
                description = ""
        if not description:
            for selector in ['meta[name="description"]', 'meta[property="og:description"]']:
                tag = soup.select_one(selector)
                if tag and tag.get("content"):
                    description = tag["content"]
                    break
        description = re.sub(r"\s+", " ", description).strip() or search_item["snippet"]
        page_text = " ".join([title, company, location, employment_type, description, search_item["snippet"]])
        page_text = re.sub(r"\s+", " ", page_text).strip()
        base.update({
            "job_title": title or search_item["title"], "company": company,
            "job_location": location, "date_posted": date_posted,
            "employment_type": employment_type,
            "description": description[:MAX_PAGE_CHARS], "page_text": page_text[:MAX_PAGE_CHARS],
            "scrape_status": "Scraped full page" if description else "Fetched page; no readable body extracted",
            "structured_jobposting": bool(posting)
        })
        return base
    except requests.exceptions.Timeout:
        base["scrape_status"] = "Page fetch timed out; search snippet retained"
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", "HTTP error")
        base["scrape_status"] = f"Page fetch failed: {status}; search snippet retained"
    except requests.exceptions.RequestException as exc:
        base["scrape_status"] = f"Page fetch error: {str(exc)[:140]}; search snippet retained"
    except Exception as exc:
        base["scrape_status"] = f"Extraction error: {str(exc)[:140]}; search snippet retained"
    return base
JOB_SIGNAL_PATTERN = re.compile(
    r"\b(job|jobs|career|careers|vacancy|vacancies|apply|application|hiring|responsibilities|qualifications|requirements|employment|position|role)\b",
    re.I
)

def is_likely_job_page(page):
    if page["structured_jobposting"]:
        return True
    url_text = (page["url"] + " " + page["title"]).lower()
    if re.search(r"/(jobs?|careers?|positions?|vacancies)(/|[-?]|$)", url_text):
        return True
    content = page["page_text"] or ""
    return bool(JOB_SIGNAL_PATTERN.search(content)) and len(content) > 180

def make_tfidf_ranking(resume, candidates):
    if not candidates:
        return []
    documents = [resume] + [page["page_text"] for page in candidates]
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True, max_features=30000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]{1,}\b"
    )
    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix[0], matrix[1:]).ravel()
    feature_names = vectorizer.get_feature_names_out()
    resume_weights = matrix[0].toarray().ravel()
    resume_indices = np.flatnonzero(resume_weights > 0)
    ranked = []
    for idx, page in enumerate(candidates):
        page_weights = matrix[idx + 1].toarray().ravel()
        if len(resume_indices):
            total_weight = float(resume_weights[resume_indices].sum())
            matching_indices = [i for i in resume_indices if page_weights[i] > 0]
            matched_weight = float(resume_weights[matching_indices].sum()) if matching_indices else 0.0
            coverage = 100 * matched_weight / total_weight if total_weight else 0.0
        else:
            matching_indices, coverage = [], 0.0
        similarity = 100 * max(0.0, min(1.0, float(similarities[idx])))
        page["text_similarity"] = round(similarity, 1)
        page["term_coverage"] = round(coverage, 1)
        page["match_index"] = round(0.75 * similarity + 0.25 * coverage, 1)
        page["matching_terms"] = [feature_names[i] for i in matching_indices][:15]
        ranked.append(page)
    ranked.sort(key=lambda row: row["match_index"], reverse=True)
    return ranked

def make_brief_summary(page, max_chars=500):
    raw = page.get("description") or page.get("snippet") or ""
    raw = re.sub(r"\s+", " ", str(raw)).strip()
    if not raw:
        return "No readable job description was returned by the source. Open the original page for details."
    noise_terms = ("accept cookies", "cookie policy", "privacy policy", "sign in", "log in", "skip to content", "all rights reserved", "javascript is required")
    sentences = re.split(r"(?<=[.!?])\s+", raw)
    useful = [s.strip() for s in sentences if len(s.split()) >= 5 and not any(n in s.lower() for n in noise_terms)]
    summary = " ".join(useful[:4]) if useful else raw
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
    return summary

def safe_job_url(url):
    try:
        parsed = urlparse(url or "")
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False

def escape_value(value):
    return html.escape(str(value or ""), quote=True)

def build_result_html(ranked_jobs):
    if not ranked_jobs:
        return "<div class='empty-state'><h3>No job pages could be ranked</h3><p>The search index may not have returned relevant vacancy pages, or websites may have restricted crawling. Open the diagnostics below to inspect the search and scrape results. CareerMatch AI will not invent vacancies.</p></div>"
    cards = []
    for rank, page in enumerate(ranked_jobs[:30], start=1):
        title = page.get("job_title") or page.get("title") or "Job opportunity"
        company = page.get("company") or "Company not identified"
        location = page.get("job_location") or "Location not stated"
        posted = page.get("date_posted") or "Not stated"
        source = page.get("engine") or "Mwmbl search"
        status = page.get("scrape_status") or "Search snippet only"
        summary = make_brief_summary(page)
        url = page.get("url") or ""
        link = f'<a class="job-link" href="{escape_value(url)}" target="_blank" rel="noopener noreferrer">Open original job page ↗</a>' if safe_job_url(url) else "<span>Original URL unavailable</span>"
        terms = page.get("matching_terms") or []
        chips = "".join(f'<span class="term-chip">{escape_value(term)}</span>' for term in terms[:8])
        rank_label = "Top ranked result" if rank == 1 else f"Ranked result {rank}"
        cards.append(
            f'<article class="job-card {"top-job" if rank == 1 else ""}">'
            f'<div class="job-card-top"><span class="rank-label">{rank_label}</span><span class="scrape-pill">{escape_value(status)}</span></div>'
            f'<h3>{escape_value(title)}</h3>'
            f'<div class="job-meta"><strong>{escape_value(company)}</strong><span>·</span><span>{escape_value(location)}</span></div>'
            f'<p class="job-summary">{escape_value(summary)}</p>'
            f'<div class="job-tags">{chips}</div>'
            f'<div class="job-footer"><span>Posted: {escape_value(posted)}</span><span>Source: {escape_value(source)}</span></div>'
            f'<div class="job-action">{link}</div></article>'
        )
    return "<div class='job-results-grid'>" + "".join(cards) + "</div>"

def build_reports(ranked_jobs, search_results, scrape_diagnostics, search_diagnostics):
    output_dir = tempfile.mkdtemp(prefix="careermatch_ai_")
    records, html_cards = [], []
    md_lines = [
        "# CareerMatch AI — Jobs Found on the Web", "",
        f"Search date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Job pages discovered: {len(ranked_jobs)}", ""
    ]
    for page in ranked_jobs:
        title = page.get("job_title") or page.get("title") or "Job opportunity"
        company = page.get("company") or "Company not identified"
        location = page.get("job_location") or "Location not stated"
        posted = page.get("date_posted") or "Not stated"
        source = page.get("engine") or "Mwmbl search"
        status = page.get("scrape_status") or "Search snippet only"
        summary = make_brief_summary(page)
        url = page.get("url") or ""
        records.append({
            "job_title": title, "company": company, "location": location,
            "brief_description": summary, "responsibilities_or_requirements": "",
            "posting_date": posted, "search_source": source,
            "page_status": status, "original_job_url": url
        })
        link_html = f'<a href="{escape_value(url)}" target="_blank" rel="noopener noreferrer">Open original job page</a>' if safe_job_url(url) else "Original URL unavailable"
        html_cards.append(
            "<article class='job-card'>"
            f"<h2>{escape_value(title)}</h2><p><b>{escape_value(company)}</b> · {escape_value(location)}</p>"
            f"<p>{escape_value(summary)}</p><p class='meta'>Posted: {escape_value(posted)} · Source: {escape_value(source)} · Page status: {escape_value(status)}</p>"
            f"<p>{link_html}</p></article>"
        )
        md_lines.extend([
            f"## {title}", "", f"**Company:** {company}  ", f"**Location:** {location}  ",
            f"**Posting date:** {posted}  ", f"**Source:** {source}  ", f"**Page status:** {status}", "",
            summary, "", f"[Open original job page]({url})" if safe_job_url(url) else "Original URL unavailable", "", "---", ""
        ])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_html = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CareerMatch AI — Job Links</title><style>body{{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;padding:0 18px;color:#172b4d;background:#f7f9fc}}.job-card{{background:#fff;border:1px solid #d0d7de;border-radius:12px;padding:18px;margin:14px 0}}.meta{{color:#667085;font-size:13px}}a{{color:#175cd3;font-weight:700;text-decoration:none}}</style></head><body><h1>CareerMatch AI — Job Opportunities</h1><p>Generated: {generated_at} · {len(records)} job pages discovered. Open each original posting to verify availability and apply.</p>{''.join(html_cards) if html_cards else '<p>No job pages were found. Review the search and scraping diagnostics.</p>'}</body></html>"""
    export_columns = ["job_title", "company", "location", "brief_description", "responsibilities_or_requirements", "posting_date", "search_source", "page_status", "original_job_url"]
    paths = {}
    paths["report_html"] = os.path.join(output_dir, "CareerMatch_Job_Links_and_Briefs.html")
    paths["report_csv"] = os.path.join(output_dir, "CareerMatch_Job_Links_and_Briefs.csv")
    paths["report_md"] = os.path.join(output_dir, "CareerMatch_Job_Links_and_Briefs.md")
    Path(paths["report_html"]).write_text(report_html, encoding="utf-8")
    pd.DataFrame(records, columns=export_columns).to_csv(paths["report_csv"], index=False)
    Path(paths["report_md"]).write_text("\n".join(md_lines), encoding="utf-8")

    paths["search_csv"] = os.path.join(output_dir, "CareerMatch_Web_Search_Results.csv")
    paths["scrape_csv"] = os.path.join(output_dir, "CareerMatch_Scrape_Diagnostics.csv")
    paths["ranked_csv"] = os.path.join(output_dir, "CareerMatch_Ranked_Scraped_Jobs.csv")
    pd.DataFrame(search_results).to_csv(paths["search_csv"], index=False)
    pd.DataFrame(scrape_diagnostics).to_csv(paths["scrape_csv"], index=False)
    ranked_export = []
    for rank, page in enumerate(ranked_jobs, start=1):
        ranked_export.append({
            "rank": rank, "job_title": page.get("job_title", ""), "company": page.get("company", ""),
            "location": page.get("job_location", ""), "date_posted": page.get("date_posted", ""),
            "employment_type": page.get("employment_type", ""), "match_index": page.get("match_index", 0),
            "text_similarity": page.get("text_similarity", 0), "term_coverage": page.get("term_coverage", 0),
            "structured_jobposting": page.get("structured_jobposting", False),
            "scrape_status": page.get("scrape_status", ""), "search_query": page.get("query", ""),
            "search_engine_source": page.get("engine", ""), "source_url": page.get("url", ""),
            "page_text_preview": (page.get("description") or page.get("snippet") or "")[:1000]
        })
    ranked_columns = ["rank", "job_title", "company", "location", "date_posted", "employment_type", "match_index", "text_similarity", "term_coverage", "structured_jobposting", "scrape_status", "search_query", "search_engine_source", "source_url", "page_text_preview"]
    pd.DataFrame(ranked_export, columns=ranked_columns).to_csv(paths["ranked_csv"], index=False)
    return paths

def run_pipeline(filename, resume_bytes, progress_callback=None):
    """Run the uploaded notebook's resume -> web search -> scrape -> TF-IDF pipeline."""
    def update(value, label):
        if progress_callback:
            progress_callback(float(value), str(label))

    output_dir = None
    try:
        update(0.03, "Reading your resume")
        extension = os.path.splitext(filename.lower())[1]
        if extension not in {".pdf", ".docx", ".txt"}:
            raise ValueError("Unsupported file format. Upload a PDF, DOCX, or TXT resume.")
        resume_text = extract_resume_text(filename, resume_bytes)
        if len(resume_text) < 100:
            raise ValueError("Very little text was extracted. This may be a scanned PDF. Use a text-based PDF, DOCX, or TXT resume.")

        update(0.12, "Extracting skills and target roles")
        resume_skills, fallback_terms, target_roles, resume_terms = build_resume_signals(resume_text)
        search_queries = make_search_queries(target_roles, resume_terms)
        if not search_queries:
            raise ValueError("No web-search queries could be generated from this resume.")

        update(0.18, "Searching the live web")
        session = make_search_session()
        search_diagnostics, search_results = [], []
        for query_index, query in enumerate(search_queries, start=1):
            results = mwmbl_search(query, session, search_diagnostics)
            if results:
                for item in results[:MAX_RESULTS_PER_QUERY]:
                    if not isinstance(item, dict):
                        continue
                    search_results.append({
                        "query": query, "title": item.get("title") or "", "url": item.get("url") or "",
                        "snippet": item.get("content") or "", "engine": item.get("engine") or "mwmbl",
                        "search_score": item.get("score")
                    })
            update(0.18 + 0.30 * query_index / len(search_queries), f"Web search {query_index}/{len(search_queries)}")
            if query_index < len(search_queries):
                time.sleep(SEARCH_DELAY_SECONDS)

        unique_search_results = {}
        for item in search_results:
            url = item["url"].strip()
            if url.startswith(("http://", "https://")):
                unique_search_results[url] = item
        search_results = list(unique_search_results.values())

        update(0.52, "Checking job pages and scraping where permitted")
        robots_cache, last_request_by_domain = {}, defaultdict(float)
        scrape_diagnostics, scraped_pages = [], []
        pages_to_scrape = search_results[:MAX_PAGES_TO_SCRAPE]
        for number, item in enumerate(pages_to_scrape, start=1):
            page = scrape_job_page(item, session, robots_cache, last_request_by_domain)
            scraped_pages.append(page)
            scrape_diagnostics.append({
                "url": item["url"], "domain": get_domain(item["url"]),
                "status": page["scrape_status"], "structured_jobposting": page["structured_jobposting"]
            })
            update(0.52 + 0.30 * number / max(1, len(pages_to_scrape)), f"Checking page {number}/{len(pages_to_scrape)}")

        update(0.85, "Filtering and ranking job pages")
        job_pages = [page for page in scraped_pages if is_likely_job_page(page)]
        deduped_pages = {}
        for page in job_pages:
            canonical = page["url"].split("#")[0].rstrip("/")
            deduped_pages[canonical] = page
        job_pages = list(deduped_pages.values())
        ranked_jobs = make_tfidf_ranking(resume_text, job_pages)

        update(0.92, "Preparing reports")
        paths = build_reports(ranked_jobs, search_results, scrape_diagnostics, search_diagnostics)
        output_dir = os.path.dirname(paths["report_html"])
        memory_zip = io.BytesIO()
        with zipfile.ZipFile(memory_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for key, path in paths.items():
                if os.path.isfile(path):
                    archive.write(path, arcname=os.path.basename(path))

        full_page_count = sum(1 for item in scraped_pages if item["scrape_status"] == "Scraped full page")
        structured_count = sum(1 for item in scraped_pages if item["structured_jobposting"])
        error_count = sum(1 for item in search_diagnostics if item.get("error"))
        diagnostics_lines = [
            f"Resume: {filename} ({len(resume_text):,} extracted characters)",
            f"Target roles detected: {', '.join(target_roles) if target_roles else 'None stated'}",
            f"Pages with full text extracted: {full_page_count}",
            f"Pages with structured JobPosting data: {structured_count}",
            f"Likely job pages after filtering: {len(job_pages)}", "",
            "Generated search queries:"
        ]
        diagnostics_lines.extend([f"- {query}" for query in search_queries])
        diagnostics_lines.extend(["", "Search API diagnostics:"])
        for item in search_diagnostics:
            suffix = f" — {item['error']}" if item.get("error") else ""
            diagnostics_lines.append(f"- {item['query']}: HTTP {item.get('http_status')} · {item.get('results', 0)} results{suffix}")
        if not search_diagnostics:
            diagnostics_lines.append("- No search response was received.")
        diagnostics_lines.extend(["", f"Search errors: {error_count}", f"Scrape attempts: {len(scrape_diagnostics)}"])
        if scrape_diagnostics:
            diagnostics_lines.extend(["", "Scrape outcomes:"])
            diagnostics_lines.extend([f"- {item['domain']}: {item['status']}" for item in scrape_diagnostics])

        results = []
        for position, page in enumerate(ranked_jobs[:30], start=1):
            results.append({
                "rank": position,
                "title": page.get("job_title") or page.get("title") or "Job opportunity",
                "company": page.get("company") or "Company not identified",
                "location": page.get("job_location") or "Location not stated",
                "posted": page.get("date_posted") or "Not stated",
                "source": page.get("engine") or "Mwmbl search",
                "status": page.get("scrape_status") or "Search snippet only",
                "summary": make_brief_summary(page),
                "url": page.get("url") or "",
                "matching_terms": page.get("matching_terms") or [],
            })
        update(1.0, "Complete")
        return {
            "results": results,
            "metrics": {
                "queries": len(search_queries), "web_results": len(search_results),
                "pages_checked": len(scraped_pages), "ranked_jobs": len(ranked_jobs),
                "full_pages": full_page_count, "structured_postings": structured_count
            },
            "resume_signals": resume_terms[:24],
            "target_roles": target_roles,
            "diagnostics": "\n".join(diagnostics_lines),
            "zip_bytes": memory_zip.getvalue(),
        }
    finally:
        if output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
