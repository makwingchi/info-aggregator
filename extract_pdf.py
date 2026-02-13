#!/usr/bin/env python3
"""
PDF Text Extraction Script
Extracts text content from a PDF file and saves it to a text file.
"""

import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Error: pdfplumber is not installed.")
    print("Please install it using: pip install pdfplumber")
    sys.exit(1)


def extract_pdf_text(pdf_path: str, output_path: str = None) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path for output file. If None, uses PDF name with .txt extension

    Returns:
        Extracted text content
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Determine output path
    if output_path is None:
        output_path = pdf_file.stem + "_extracted.txt"

    print(f"Extracting text from: {pdf_file.name}")

    # Extract text from PDF
    extracted_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}")

        for i, page in enumerate(pdf.pages, 1):
            print(f"Processing page {i}/{total_pages}...", end='\r')
            text = page.extract_text()
            if text:
                extracted_text.append(f"\n{'='*80}\n")
                extracted_text.append(f"PAGE {i}\n")
                extracted_text.append(f"{'='*80}\n\n")
                extracted_text.append(text)
                extracted_text.append("\n")

    print(f"\nExtraction complete!")

    # Combine all text
    full_text = "".join(extracted_text)

    # Save to file
    output_file = Path(output_path)
    output_file.write_text(full_text, encoding='utf-8')
    print(f"Saved to: {output_file.absolute()}")

    return full_text


def main():
    # PDF path from the user's request
    pdf_path = "/Users/edy/Zotero/storage/XTMKK42P/Pourreza 等 - 2024 - CHASE-SQL Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL.pdf"

    # Output file in current directory
    output_path = "CHASE_SQL_extracted.txt"

    try:
        extract_pdf_text(pdf_path, output_path)
        print("\n✓ Text extraction successful!")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
