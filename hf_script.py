from datetime import date, timedelta
from urllib.request import Request, urlopen

yesterday = (date.today() - timedelta(days=1)).isoformat()
url = f"https://huggingface.co/papers/date/{yesterday}"
req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urlopen(req, timeout=30).read().decode("utf-8", errors="replace")

with open("huggingface.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved huggingface.html for", yesterday)