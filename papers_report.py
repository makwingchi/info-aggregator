#!/usr/bin/env python3
"""
Scrape AlphaXiv and Hugging Face papers, dedupe across days/sources,
download PDFs, and summarize with an LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from html import unescape
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from loguru import logger

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None


ALPHAXIV_URL = "https://www.alphaxiv.org/?sort=Likes&interval=7+Days"
HF_URL_TEMPLATE = "https://huggingface.co/papers/date/{date}"
ARXIV_PDF_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

DEFAULT_TZ = "UTC"
DEFAULT_LIKES_THRESHOLD = 50
DEFAULT_HF_LIKES_THRESHOLD = 40
ENV_KEYS_AZURE = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z\-]+/\d{7})(v\d+)?", re.I)
HF_PAPER_RE = re.compile(r'href="/papers/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?"')
NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S | re.I
)
DATA_PROPS_RE = re.compile(
    r'data-target="DailyPapers"[^>]+data-props="(.*?)"', re.S | re.I
)


def fetch_url(url: str, timeout: int = 30) -> str:
    # Fetch remote HTML/JSON with a consistent user agent.
    logger.debug(f"Fetching URL: {url} (timeout={timeout})")
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; papers-report/1.0)",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def strip_tags(text: str) -> str:
    # Remove HTML tags and normalize whitespace.
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return " ".join(unescape(cleaned).split())


def normalize_arxiv_id(text: Any) -> Optional[str]:
    # Extract and normalize an arXiv ID from a blob of text.
    if not text:
        return None
    match = ARXIV_ID_RE.search(str(text))
    if not match:
        return None
    if match.group(1):
        return match.group(1)
    if match.group(3):
        return match.group(3)
    return None


def extract_likes(text: str) -> Optional[int]:
    # Find a likes count inside a text snippet.
    if not text:
        return None
    patterns = [
        r"data-likes=\"(\d+)\"",
        r"\bLikes?\b[^0-9]{0,6}(\d+)",
        r"\b(\d+)\s*Likes?\b",
        r"\b(\d+)\s*likes?\b",
        r"\b(\d+)\s*❤",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.I)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def extract_json_from_next_data(html: str) -> Optional[Dict[str, Any]]:
    # Parse __NEXT_DATA__ JSON payload if present.
    match = NEXT_DATA_RE.search(html)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse __NEXT_DATA__: {exc}")
        return None


def extract_json_from_data_props(html: str) -> Optional[Dict[str, Any]]:
    # Parse data-props JSON payload used by Hugging Face DailyPapers.
    match = DATA_PROPS_RE.search(html)
    if not match:
        return None
    try:
        payload = unescape(match.group(1))
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse data-props JSON: {exc}")
        return None


def walk_json(obj: Any) -> Iterable[Dict[str, Any]]:
    # Yield dict nodes from nested JSON.
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_json(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_json(item)


def collect_papers_from_json(data: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    # Heuristically extract paper-like dicts from JSON payloads.
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for node in walk_json(data):
        # For Hugging Face, prefer the nested 'paper' object if it exists
        paper_obj = node.get("paper")
        paper_node = paper_obj if isinstance(paper_obj, dict) else node

        title = paper_node.get("title") or paper_node.get("paper_title") or node.get("title")
        arxiv_id = (
            paper_node.get("arxivId")
            or paper_node.get("arxiv_id")
            or paper_node.get("arxiv")
            or normalize_arxiv_id(paper_node.get("id") or paper_node.get("url") or "")
        )
        if not arxiv_id:
            arxiv_id = normalize_arxiv_id(json.dumps(paper_node)[:200])
        if not title or not arxiv_id:
            continue
        arxiv_id = normalize_arxiv_id(arxiv_id)
        if not arxiv_id or arxiv_id in seen:
            continue
        likes = None
        for key in ("upvotes", "likes", "likeCount", "like_count", "likesCount"):
            if isinstance(paper_node, dict) and key in paper_node:
                try:
                    likes = int(paper_node[key])
                except (TypeError, ValueError):
                    likes = None
                break

        # Use arXiv URL as default if not present
        url = paper_node.get("url") or node.get("url")
        if not url and arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        results.append(
            {
                "source": source,
                "title": " ".join(str(title).split()),
                "arxiv_id": arxiv_id,
                "url": url,
                "likes": likes,
            }
        )
        seen.add(arxiv_id)
    return results


def parse_alphaxiv_html(html: str) -> List[Dict[str, Any]]:
    # Extract papers from AlphaXiv HTML when JSON is unavailable.
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()
    title_pattern = re.compile(
        r'<a[^>]+href="/abs/(?P<id>\d{4}\.\d{4,5})"[^>]*>.*?'
        r'<div[^>]*text-\[22px\][^>]*>(?P<title>.*?)</div>',
        re.S | re.I,
    )
    matches = list(title_pattern.finditer(html))
    for idx, match in enumerate(matches):
        arxiv_id = normalize_arxiv_id(match.group("id"))
        if not arxiv_id or arxiv_id in seen:
            continue
        title = strip_tags(match.group("title"))
        segment_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(html)
        segment = html[match.start() : segment_end]
        likes_match = re.search(
            r"lucide-thumbs-up.*?<span>([\d,]+)</span>", segment, re.S | re.I
        )
        likes = None
        if likes_match:
            try:
                likes = int(likes_match.group(1).replace(",", ""))
            except (TypeError, ValueError):
                likes = None
        results.append(
            {
                "source": "alphaxiv",
                "title": title or None,
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "likes": likes,
            }
        )
        seen.add(arxiv_id)
    if results:
        return results
    # Fallback: attempt a generic arXiv ID scan if the card parsing fails.
    for match in ARXIV_ID_RE.finditer(html):
        arxiv_id = normalize_arxiv_id(match.group(0))
        if not arxiv_id or arxiv_id in seen:
            continue
        start = max(0, match.start() - 500)
        end = min(len(html), match.end() + 500)
        context = html[start:end]
        title_match = re.search(r"<h[23][^>]*>(.*?)</h[23]>", context, re.S | re.I)
        title = strip_tags(title_match.group(1)) if title_match else None
        likes_match = re.search(
            r"class=['\"]leading-none['\"][^>]*>\s*([0-9,]+)\s*<",
            context,
            re.S | re.I,
        )
        if likes_match:
            try:
                likes = int(likes_match.group(1).replace(",", ""))
            except (TypeError, ValueError):
                likes = None
        else:
            likes = extract_likes(context)
        results.append(
            {
                "source": "alphaxiv",
                "title": title,
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "likes": likes,
            }
        )
        seen.add(arxiv_id)
    return results


def parse_hf_html(html: str) -> List[Dict[str, Any]]:
    # Extract papers from Hugging Face HTML when JSON is unavailable.
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for match in HF_PAPER_RE.finditer(html):
        arxiv_id = normalize_arxiv_id(match.group(1))
        if not arxiv_id or arxiv_id in seen:
            continue
        start = max(0, match.start() - 1500)
        end = min(len(html), match.end() + 400)
        context = html[start:end]
        title_match = re.search(r"<h[23][^>]*>(.*?)</h[23]>", context, re.S | re.I)
        title = strip_tags(title_match.group(1)) if title_match else None

        # Look for likes in <div class="leading-none">NUMBER</div> pattern
        likes_match = re.search(
            r'<div\s+class="leading-none">(\d+)</div>',
            context,
            re.S | re.I
        )
        if likes_match:
            try:
                likes = int(likes_match.group(1))
            except (TypeError, ValueError):
                likes = extract_likes(context)
        else:
            likes = extract_likes(context)

        results.append(
            {
                "source": "huggingface",
                "title": title,
                "arxiv_id": arxiv_id,
                "url": f"https://huggingface.co/papers/{arxiv_id}",
                "likes": likes,
            }
        )
        seen.add(arxiv_id)
    return results


def parse_alphaxiv(html: str) -> List[Dict[str, Any]]:
    # Parse AlphaXiv using JSON payloads with an HTML fallback.
    data = extract_json_from_next_data(html)
    if data:
        papers = collect_papers_from_json(data, "alphaxiv")
        if papers:
            return papers
    return parse_alphaxiv_html(html)


def parse_huggingface(html: str) -> List[Dict[str, Any]]:
    # Parse Hugging Face using JSON payloads with an HTML fallback.
    data = extract_json_from_next_data(html)
    if data:
        papers = collect_papers_from_json(data, "huggingface")
        if papers:
            return papers
    data = extract_json_from_data_props(html)
    if data:
        papers = collect_papers_from_json(data, "huggingface")
        if papers:
            return papers
    return parse_hf_html(html)


def load_previous_ids(path: str) -> set[str]:
    # Load previously scraped arXiv IDs for deduping.
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to load {path}: {exc}")
        return set()
    items = payload.get("items", payload if isinstance(payload, list) else [])
    result = set()
    for item in items:
        arxiv_id = item.get("arxiv_id")
        if arxiv_id:
            result.add(arxiv_id)
    return result


def save_payload(path: str, payload: Dict[str, Any]) -> None:
    # Persist scraped metadata to JSON.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def filter_alphaxiv_by_likes(
    papers: Sequence[Dict[str, Any]], min_likes: int
) -> List[Dict[str, Any]]:
    # Keep AlphaXiv papers that meet the likes threshold.
    filtered = []
    for paper in papers:
        likes = paper.get("likes")
        if likes is None:
            continue
        try:
            likes_value = int(likes)
        except (TypeError, ValueError):
            continue
        if likes_value >= min_likes:
            paper["likes"] = likes_value
            filtered.append(paper)
    return filtered


def filter_huggingface_by_likes(
    papers: Sequence[Dict[str, Any]], min_likes: int
) -> List[Dict[str, Any]]:
    # Keep Hugging Face papers whose likes exceed the threshold.
    filtered = []
    for paper in papers:
        likes = paper.get("likes")
        if likes is None:
            continue
        try:
            likes_value = int(likes)
        except (TypeError, ValueError):
            continue
        if likes_value > min_likes:
            paper["likes"] = likes_value
            filtered.append(paper)
    return filtered


def dedupe_by_ids(
    papers: Sequence[Dict[str, Any]], seen_ids: set[str]
) -> List[Dict[str, Any]]:
    # Drop papers whose arXiv IDs appear in the seen set.
    result = []
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id or arxiv_id in seen_ids:
            continue
        result.append(paper)
    return result


def merge_sources(papers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Merge papers across sources using arXiv ID as the key.
    merged: Dict[str, Dict[str, Any]] = {}
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        entry = merged.get(arxiv_id)
        if not entry:
            entry = dict(paper)
            entry["sources"] = sorted({paper.get("source")} - {None})
            merged[arxiv_id] = entry
            continue
        entry["sources"] = sorted(set(entry.get("sources", [])) | {paper.get("source")})
        if not entry.get("title") and paper.get("title"):
            entry["title"] = paper["title"]
        if not entry.get("url") and paper.get("url"):
            entry["url"] = paper["url"]
        if paper.get("likes") is not None:
            entry["likes"] = max(entry.get("likes") or 0, paper["likes"])
    return list(merged.values())


def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    # Yield chunks of a list.
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def fetch_arxiv_metadata(arxiv_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    # Fetch title/abstract/authors from the arXiv API.
    results: Dict[str, Dict[str, Any]] = {}
    if not arxiv_ids:
        return results
    atom = "{http://www.w3.org/2005/Atom}"
    for batch in chunked(list(arxiv_ids), 20):
        params = urlencode({"id_list": ",".join(batch), "max_results": str(len(batch))})
        url = f"{ARXIV_API_URL}?{params}"
        try:
            xml_text = fetch_url(url, timeout=30)
        except Exception as exc:
            logger.warning(f"Failed to fetch arXiv metadata: {exc}")
            continue
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning(f"Failed to parse arXiv metadata: {exc}")
            continue
        for entry in root.findall(f"{atom}entry"):
            id_text = entry.findtext(f"{atom}id", default="")
            arxiv_id = normalize_arxiv_id(id_text)
            if not arxiv_id:
                continue
            title = entry.findtext(f"{atom}title", default="")
            summary = entry.findtext(f"{atom}summary", default="")
            authors = [
                author.findtext(f"{atom}name", default="")
                for author in entry.findall(f"{atom}author")
            ]
            results[arxiv_id] = {
                "title": " ".join(title.split()),
                "abstract": " ".join(summary.split()),
                "authors": [name for name in authors if name],
            }
        time.sleep(0.5)
    return results


def safe_arxiv_filename(arxiv_id: str) -> str:
    # Build a filesystem-friendly PDF name.
    return arxiv_id.replace("/", "_") + ".pdf"


def download_pdf(arxiv_id: str, output_dir: str) -> Optional[str]:
    # Download an arXiv PDF if not already present.
    if not arxiv_id:
        return None
    os.makedirs(output_dir, exist_ok=True)
    filename = safe_arxiv_filename(arxiv_id)
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        return path
    url = ARXIV_PDF_TEMPLATE.format(arxiv_id=arxiv_id)
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; papers-report/1.0)",
            },
        )
        with urlopen(req, timeout=300) as resp, open(path, "wb") as handle:
            handle.write(resp.read())
        return path
    except Exception as exc:
        logger.warning(f"Failed to download {arxiv_id}: {exc}")
        return None


def build_llm_client() -> Tuple[str, Any, str]:
    # Build either Azure OpenAI or OpenAI client config.
    missing_azure = [key for key in ENV_KEYS_AZURE if not os.getenv(key)]
    if not missing_azure:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install the openai package (pip install openai)."
            ) from exc
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
        return "azure", client, os.environ["AZURE_OPENAI_DEPLOYMENT"]
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install the openai package (pip install openai)."
            ) from exc
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return "openai", OpenAI(api_key=os.environ["OPENAI_API_KEY"]), model
    missing_text = ", ".join(missing_azure)
    raise RuntimeError(
        "Missing LLM config. Provide Azure OpenAI env vars or OPENAI_API_KEY. "
        f"Missing Azure vars: {missing_text}"
    )


def extract_text_from_pdf(pdf_path: str, max_pages: int = 200) -> Optional[str]:
    """Extract text from PDF file using PyPDF2."""
    try:
        import PyPDF2
    except ImportError:
        logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
        return None

    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = min(len(pdf_reader.pages), max_pages)

            text_parts = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            full_text = "\n\n".join(text_parts)

            # Limit text length to avoid token limits (roughly 100k characters = ~25k tokens)
            if len(full_text) > 1_000_000:
                full_text = full_text[:1000000] + "\n\n[Text truncated due to length...]"

            return full_text
    except Exception as exc:
        logger.warning(f"Failed to extract text from PDF {pdf_path}: {exc}")
        return None


def summarize_paper_llm(
    client_info: Tuple[str, Any, str], paper: Dict[str, Any]
) -> str:
    # Summarize a paper using LLM with full PDF text if available.
    provider, client, model = client_info
    arxiv_id = paper.get("arxiv_id")
    pdf_path = paper.get("pdf_path")

    system = (
        "Reply as an expert in Large Language Model and Agent Application. "
        "Make an academic document of the paper highlighting direct important quotes and ideas. "
        "Be in depth. Make no use of bullet points, and add direct quotes from the text. "
        "Think in English, and respond in Simplified Chinese."
    )

    # Build user message with metadata
    metadata = {
        "title": paper.get("title"),
        "authors": paper.get("authors"),
        "abstract": paper.get("abstract"),
        "arxiv_id": arxiv_id,
        "sources": paper.get("sources"),
        "likes": paper.get("likes"),
    }

    user_content = f"Please analyze this paper:\n\n{json.dumps(metadata, ensure_ascii=True, indent=2)}"

    # Extract and include PDF text if available
    pdf_text = None
    if pdf_path and os.path.exists(pdf_path):
        logger.debug(f"Extracting text from PDF: {pdf_path}")
        pdf_text = extract_text_from_pdf(pdf_path)

    if pdf_text:
        user_content += f"\n\n=== FULL PAPER TEXT ===\n\n{pdf_text}"
        user_content += "\n\nPlease provide a comprehensive analysis with direct quotes from the paper text above."
    else:
        user_content += "\n\nNote: PDF text extraction failed or not available. Analysis based on abstract and metadata only."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    if provider == "azure":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    return response.choices[0].message.content.strip()


def render_report(
    report_date: date,
    alpha_date: date,
    hf_date: date,
    alpha_stats: Dict[str, int],
    hf_stats: Dict[str, int],
    duplicates: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
) -> str:
    # Build a markdown report for the run.
    lines = [f"# Papers Report ({report_date.isoformat()})", ""]

    lines.append("## Summary")
    lines.append(
        f"- **Total papers**: {len(papers)} "
        f"({alpha_stats['new']} from AlphaXiv, {hf_stats['new']} from HuggingFace)"
    )
    lines.append(f"- **Cross-source duplicates**: {len(duplicates)}")
    lines.append("")

    lines.append("## Source Statistics")
    lines.append(
        f"### AlphaXiv ({alpha_date.isoformat()})"
    )
    lines.append(f"- Scraped: {alpha_stats['raw']} papers")
    lines.append(f"- Filtered (likes ≥ {alpha_stats['threshold']}): {alpha_stats['filtered']} papers")
    lines.append(f"- New (not in yesterday's data): **{alpha_stats['new']} papers**")
    lines.append("")

    lines.append(
        f"### Hugging Face Daily Papers ({hf_date.isoformat()})"
    )
    lines.append(f"- Scraped: {hf_stats['raw']} papers")
    lines.append(f"- Filtered (likes > {hf_stats['threshold']}): {hf_stats['filtered']} papers")
    lines.append(f"- New (not in AlphaXiv): **{hf_stats['new']} papers**")
    lines.append("")

    if duplicates:
        lines.append("## Cross-Source Duplicates")
        lines.append("Papers that appear in both AlphaXiv and Hugging Face:")
        lines.append("")
        for dup in duplicates:
            title = dup.get("title") or "Untitled"
            arxiv_id = dup.get("arxiv_id") or "unknown"
            sources = ", ".join(dup.get("sources", []))
            lines.append(f"- **{title}**")
            lines.append(f"  - arXiv: [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
            lines.append(f"  - Sources: {sources}")
        lines.append("")

    lines.append("## Papers")
    if not papers:
        lines.append("No papers found matching the criteria.")
    else:
        for idx, paper in enumerate(papers, 1):
            title = paper.get("title") or "Untitled"
            arxiv_id = paper.get("arxiv_id") or "unknown"
            sources = ", ".join(paper.get("sources") or [])
            likes = paper.get("likes")
            authors = paper.get("authors", [])

            lines.append(f"### {idx}. {title}")
            lines.append(f"- **arXiv**: [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
            if sources:
                lines.append(f"- **Sources**: {sources}")
            if likes is not None:
                lines.append(f"- **Likes**: {likes}")
            if authors:
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += f" et al. ({len(authors)} authors)"
                lines.append(f"- **Authors**: {author_str}")

            pdf_path = paper.get("pdf_path")
            if pdf_path:
                lines.append(f"- **PDF**: `{pdf_path}`")

            # Add abstract
            abstract = paper.get("abstract")
            if abstract:
                lines.append("")
                lines.append("**Abstract:**")
                lines.append(abstract)

            summary = paper.get("summary")
            if summary:
                lines.append("")
                lines.append("**Summary:**")
                lines.append(summary)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def resolve_date(date_str: Optional[str], tz_name: str) -> date:
    # Resolve a date string or default to today in the requested timezone.
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    if ZoneInfo is not None:
        now = datetime.now(ZoneInfo(tz_name))
    else:
        now = datetime.now()
    return now.date()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape papers from AlphaXiv and Hugging Face, deduplicate, and summarize with LLM."
    )
    parser.add_argument(
        "--date",
        help="Date to treat as today (YYYY-MM-DD). AlphaXiv uses this date, HF uses yesterday.",
    )
    parser.add_argument("--tz", default=DEFAULT_TZ, help="Timezone for date resolution.")
    parser.add_argument(
        "--likes-threshold",
        type=int,
        default=DEFAULT_LIKES_THRESHOLD,
        help="Minimum AlphaXiv likes to include (>=).",
    )
    parser.add_argument(
        "--hf-likes-threshold",
        type=int,
        default=DEFAULT_HF_LIKES_THRESHOLD,
        help="Minimum Hugging Face likes to include (>).",
    )
    parser.add_argument("--data-dir", default="data", help="Directory for scraped JSON.")
    parser.add_argument("--pdf-dir", default="papers", help="Directory for PDFs.")
    parser.add_argument("--report-dir", default="reports", help="Directory for markdown reports.")
    parser.add_argument("--env-file", default=".env", help="Path to dotenv file.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM summaries.")
    parser.add_argument("--max-papers", type=int, default=50, help="Limit number of papers.")
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)

    today = resolve_date(args.date, args.tz)
    alpha_date = today
    hf_date = today - timedelta(days=1)
    alpha_prev_date = alpha_date - timedelta(days=1)

    logger.info(f"Starting paper scraping for {today.isoformat()}")
    logger.info(f"AlphaXiv date: {alpha_date.isoformat()}, HuggingFace date: {hf_date.isoformat()}")

    # Step 1: Scrape AlphaXiv
    logger.info("Step 1/6: Scraping AlphaXiv...")
    alpha_html = fetch_url(ALPHAXIV_URL)
    alpha_raw = parse_alphaxiv(alpha_html)
    logger.info(f"  Found {len(alpha_raw)} papers from AlphaXiv")

    alpha_filtered = filter_alphaxiv_by_likes(alpha_raw, args.likes_threshold)
    logger.info(f"  Filtered to {len(alpha_filtered)} papers with likes >= {args.likes_threshold}")

    # Step 2: Deduplicate AlphaXiv against yesterday
    logger.info("Step 2/6: Deduplicating AlphaXiv against yesterday's data...")
    alpha_prev_path = os.path.join(args.data_dir, f"alphaxiv_{alpha_prev_date:%Y%m%d}.json")
    alpha_seen = load_previous_ids(alpha_prev_path)
    alpha_new = dedupe_by_ids(alpha_filtered, alpha_seen)
    logger.info(f"  Found {len(alpha_new)} new papers (not in yesterday's {len(alpha_seen)} papers)")

    alpha_payload = {
        "date": alpha_date.isoformat(),
        "source": "alphaxiv",
        "items": alpha_filtered,
    }
    alpha_path = os.path.join(args.data_dir, f"alphaxiv_{alpha_date:%Y%m%d}.json")
    save_payload(alpha_path, alpha_payload)
    logger.info(f"  Saved AlphaXiv data to {alpha_path}")

    # Step 3: Scrape Hugging Face
    logger.info("Step 3/6: Scraping Hugging Face daily papers...")
    hf_url = HF_URL_TEMPLATE.format(date=hf_date.isoformat())
    logger.info(f"  URL: {hf_url}")
    hf_html = fetch_url(hf_url)
    hf_raw = parse_huggingface(hf_html)
    logger.info(f"  Found {len(hf_raw)} papers from Hugging Face")

    hf_filtered = filter_huggingface_by_likes(hf_raw, args.hf_likes_threshold)
    logger.info(f"  Filtered to {len(hf_filtered)} papers with likes > {args.hf_likes_threshold}")

    # Step 4: Deduplicate HF against AlphaXiv (cross-source dedup)
    logger.info("Step 4/6: Deduplicating Hugging Face against AlphaXiv...")
    alpha_ids = {paper.get("arxiv_id") for paper in alpha_filtered if paper.get("arxiv_id")}
    hf_new = dedupe_by_ids(hf_filtered, alpha_ids)
    logger.info(f"  Found {len(hf_new)} papers unique to Hugging Face")

    hf_payload = {
        "date": hf_date.isoformat(),
        "source": "huggingface",
        "items": hf_filtered,
    }
    hf_path = os.path.join(args.data_dir, f"huggingface_{hf_date:%Y%m%d}.json")
    save_payload(hf_path, hf_payload)
    logger.info(f"  Saved Hugging Face data to {hf_path}")

    # Step 5: Merge and process
    logger.info("Step 5/6: Merging sources and downloading PDFs...")
    combined = merge_sources(alpha_new + hf_new)
    logger.info(f"  Total papers after merge: {len(combined)}")

    if args.max_papers and len(combined) > args.max_papers:
        logger.info(f"  Limiting to {args.max_papers} papers")
        combined = combined[: args.max_papers]

    arxiv_ids = [paper["arxiv_id"] for paper in combined if paper.get("arxiv_id")]
    logger.info(f"  Fetching metadata for {len(arxiv_ids)} papers from arXiv API...")
    metadata = fetch_arxiv_metadata(arxiv_ids)

    logger.info(f"  Enriched {len(metadata)} papers with arXiv metadata")

    for paper in combined:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        meta = metadata.get(arxiv_id)
        if not meta:
            continue
        paper["title"] = meta.get("title")
        paper["abstract"] = meta.get("abstract")
        paper["authors"] = meta.get("authors")

    duplicates = [paper for paper in combined if len(paper.get("sources", [])) > 1]
    logger.info(f"  Identified {len(duplicates)} cross-source duplicates")

    logger.info(f"  Downloading PDFs to {args.pdf_dir}...")
    pdf_dir = os.path.join(args.pdf_dir, today.isoformat())
    downloaded_count = 0
    for paper in combined:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        pdf_path = download_pdf(arxiv_id, pdf_dir)
        if pdf_path:
            paper["pdf_path"] = pdf_path
            downloaded_count += 1
    logger.info(f"  Downloaded {downloaded_count}/{len(combined)} PDFs")

    # Step 6: Generate LLM summaries
    logger.info("Step 6/6: Generating LLM summaries...")
    client_info = None
    if not args.no_llm and combined:
        try:
            client_info = build_llm_client()
            logger.info("  LLM client initialized successfully")
        except Exception as exc:
            logger.error(f"  Failed to initialize LLM client: {exc}")
            logger.info("  Skipping LLM summarization")

    summarized_count = 0
    for idx, paper in enumerate(combined, 1):
        if args.no_llm or not client_info:
            paper["summary"] = "(LLM summarization skipped)"
            continue
        abstract = paper.get("abstract")
        if not abstract:
            paper["summary"] = "(Missing abstract; skipping LLM summarization.)"
            continue
        try:
            logger.info(f"  Summarizing paper {idx}/{len(combined)}: {paper.get('arxiv_id')}")
            paper["summary"] = summarize_paper_llm(client_info, paper)
            summarized_count += 1
        except Exception as exc:
            logger.warning(f"  LLM summarization failed for {paper.get('arxiv_id')}: {exc}")
            paper["summary"] = "(LLM summarization failed.)"

    if not args.no_llm:
        logger.info(f"  Successfully summarized {summarized_count}/{len(combined)} papers")

    # Generate report
    logger.info("Generating markdown report...")
    report = render_report(
        report_date=today,
        alpha_date=alpha_date,
        hf_date=hf_date,
        alpha_stats={
            "raw": len(alpha_raw),
            "threshold": args.likes_threshold,
            "filtered": len(alpha_filtered),
            "new": len(alpha_new),
        },
        hf_stats={
            "raw": len(hf_raw),
            "threshold": args.hf_likes_threshold,
            "filtered": len(hf_filtered),
            "new": len(hf_new),
        },
        duplicates=duplicates,
        papers=combined,
    )

    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, f"papers_{today:%Y%m%d}.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    logger.info(f"✓ Report saved to {report_path}")
    logger.info(f"✓ Total papers: {len(combined)} ({len(alpha_new)} AlphaXiv + {len(hf_new)} HuggingFace)")
    logger.info(f"✓ Cross-source duplicates: {len(duplicates)}")
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
