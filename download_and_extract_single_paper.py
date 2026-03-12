from papers_report import extract_paper_content

# 指定单独的 Markdown 保存目录
content, md_path = extract_paper_content(
    "2503.02240v2",
    output_dir="papers",
    markdown_dir="markdowns"  # Markdown 文件保存到 markdowns/
)

if content:
    print(f"Markdown 已保存到: {md_path}")
