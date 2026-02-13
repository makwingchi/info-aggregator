# Info Aggregator

A multi-purpose data aggregation and analysis tool that collects information from diverse sources including academic papers and NBA game statistics, then generates comprehensive reports using LLM-powered summarization.

## Overview

This repository contains Python scripts for:

1. **Academic Paper Aggregation** - Scraping and analyzing research papers from AlphaXiv and Hugging Face
2. **NBA Game Analysis** - Fetching and summarizing NBA game data with momentum analysis and player highlights

## Features

### Academic Papers (`papers_report.py`)

- **Multi-Source Scraping**: Aggregates papers from AlphaXiv and Hugging Face Daily Papers
- **Smart Deduplication**: Removes duplicates across sources and time periods
- **Popularity Filtering**: Filters papers based on likes/upvotes thresholds
- **PDF Download**: Automatically downloads arXiv PDFs for selected papers
- **Full-Text Analysis**: Extracts and analyzes complete PDF content (up to 200 pages)
- **LLM Summarization**: Generates in-depth academic summaries in Simplified Chinese using Azure OpenAI or OpenAI
- **Metadata Enrichment**: Fetches complete metadata (title, abstract, authors) from arXiv API
- **Markdown Reports**: Produces structured reports with paper details and summaries

**Key Capabilities:**
- Tracks papers across days to identify new publications
- Merges cross-source duplicates intelligently
- Supports both Azure OpenAI and standard OpenAI APIs
- Configurable likes thresholds for quality filtering
- Comprehensive logging with loguru

### NBA Game Reports (`dunks_report.py`)

- **Game Data Extraction**: Parses embedded JavaScript data from Dunks & Threes website
- **Momentum Analysis**: Calculates quarter-by-quarter scoring, lead changes, and scoring runs
- **Win Probability Tracking**: Identifies biggest win probability swings
- **Player Highlights**: Extracts top performers with detailed statistics
- **Play-by-Play Processing**: Analyzes scoring plays and game flow
- **LLM Summarization**: Generates comprehensive game recaps in Simplified Chinese

**Key Capabilities:**
- Parses complex JavaScript objects from HTML
- Calculates advanced metrics (TS%, +/-, win probability)
- Supports both LLM-powered and deterministic summaries
- Timezone-aware date handling
- Caching support for offline analysis

## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/makwingchi/info-aggregator.git
cd info-aggregator
```

2. Install dependencies:
```bash
poetry install
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```
# For Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=your_deployment_name

# OR for OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

## Usage

### Academic Papers Report

Generate a daily report of trending academic papers:

```bash
poetry run python papers_report.py
```

**Options:**
```bash
--date YYYY-MM-DD          # Override date (default: today)
--tz TIMEZONE              # Timezone for date resolution (default: UTC)
--likes-threshold N        # Min AlphaXiv likes (default: 50)
--hf-likes-threshold N     # Min HuggingFace likes (default: 40)
--data-dir PATH            # Directory for scraped JSON (default: data)
--pdf-dir PATH             # Directory for PDFs (default: papers)
--report-dir PATH          # Directory for reports (default: reports)
--no-llm                   # Skip LLM summarization
--max-papers N             # Limit number of papers (default: 50)
```

**Example:**
```bash
poetry run python papers_report.py --likes-threshold 100 --max-papers 20
```

### NBA Game Report

Generate a report for yesterday's NBA games:

```bash
poetry run python dunks_report.py
```

**Options:**
```bash
--date YYYY-MM-DD          # Override date
--tz TIMEZONE              # Timezone (default: America/New_York)
--max-pbp N                # Max play-by-play events (default: 80)
--no-llm                   # Skip LLM summarization
--output PATH              # Write to file instead of stdout
--games-html PATH          # Use cached games HTML
--game-html-dir PATH       # Use cached game HTML files
```

**Example:**
```bash
poetry run python dunks_report.py --date 2024-02-10 --output report.md
```

### Utility Scripts

Quick data fetching scripts for testing:

```bash
# Fetch AlphaXiv HTML
poetry run python alphaxiv_script.py

# Fetch HuggingFace papers HTML
poetry run python hf_script.py
```

## Project Structure

```
info-aggregator/
├── papers_report.py          # Academic paper aggregation script
├── dunks_report.py           # NBA game analysis script
├── alphaxiv_script.py        # AlphaXiv HTML fetcher
├── hf_script.py              # HuggingFace HTML fetcher
├── extract_pdf.py            # PDF text extraction utility
├── scripts/
│   └── run_dunks_report.sh   # Shell script for automated runs
├── data/                     # Scraped JSON data (gitignored)
├── papers/                   # Downloaded PDFs (gitignored)
├── reports/                  # Generated markdown reports (gitignored)
├── pyproject.toml            # Poetry dependencies
├── poetry.lock               # Locked dependencies
└── .env                      # Environment variables (gitignored)
```

## Output Examples

### Papers Report Structure

```markdown
# Papers Report (2024-02-13)

## Summary
- Total papers: 15 (10 from AlphaXiv, 5 from HuggingFace)
- Cross-source duplicates: 2

## Source Statistics
### AlphaXiv (2024-02-13)
- Scraped: 150 papers
- Filtered (likes ≥ 50): 25 papers
- New (not in yesterday's data): 10 papers

### Hugging Face Daily Papers (2024-02-12)
- Scraped: 30 papers
- Filtered (likes > 40): 8 papers
- New (not in AlphaXiv): 5 papers

## Papers
### 1. Paper Title Here
- arXiv: [2402.xxxxx](https://arxiv.org/abs/2402.xxxxx)
- Sources: alphaxiv, huggingface
- Likes: 125
- Authors: Author One, Author Two, et al. (5 authors)

**Abstract:**
[Paper abstract here]

**Summary:**
[LLM-generated summary in Simplified Chinese]
```

### NBA Game Report Structure

```markdown
# NBA Games Report (2024-02-12)

## Team A at Team B (105-112)
Matchup: Team A vs Team B

- Momentum
  - largest win-prob swing: Team B +0.15 at Q4 2:34
  - largest run: Team B 12-0; lead changes: 8
  - Q1 Team A 28-25 Team B
  - Q2 Team A 22-30 Team B
  - Q3 Team A 26-28 Team B
  - Q4 Team A 29-29 Team B

- Player highlights
  - Team B: Player X (32 pts, 8 reb, 5 ast, TS 0.65, FG 12/20, 3P 4/8, FT 4/4, 2 blk, 1 stl, 2 tov, +/- +12)
  - Team A: Player Y (28 pts, 6 reb, 9 ast, TS 0.58, FG 10/22, 3P 3/9, FT 5/6, 0 blk, 3 stl, 4 tov, +/- -8)
```

## Technical Details

### Paper Aggregation Pipeline

1. **Scraping**: Fetches HTML from AlphaXiv and HuggingFace
2. **Parsing**: Extracts paper data from JSON payloads or HTML fallback
3. **Filtering**: Applies likes thresholds to identify popular papers
4. **Deduplication**: Removes papers seen in previous days and across sources
5. **Enrichment**: Fetches metadata from arXiv API
6. **PDF Download**: Downloads PDFs for selected papers
7. **Text Extraction**: Extracts full text from PDFs using PyPDF2
8. **Summarization**: Generates academic summaries using LLM
9. **Report Generation**: Creates markdown report with all data

### NBA Game Analysis Pipeline

1. **Data Fetching**: Retrieves game HTML from Dunks & Threes
2. **JavaScript Parsing**: Extracts embedded game data objects
3. **Metric Calculation**: Computes momentum signals and player stats
4. **Highlight Extraction**: Identifies top performers and key plays
5. **Summarization**: Generates game recap using LLM or deterministic method
6. **Report Generation**: Creates markdown report with analysis

## Dependencies

Core dependencies (managed by Poetry):
- `loguru` - Advanced logging
- `python-dotenv` - Environment variable management
- `openai` - OpenAI/Azure OpenAI API client
- `pypdf2` - PDF text extraction
- `pdfplumber` - Alternative PDF processing
- `ipykernel` - Jupyter notebook support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Data sources: AlphaXiv, Hugging Face, Dunks & Threes
- LLM providers: Azure OpenAI, OpenAI
- arXiv API for paper metadata

## Contact

For questions or feedback, please open an issue on GitHub.
