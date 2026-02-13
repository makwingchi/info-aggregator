from urllib.request import Request, urlopen

url = "https://www.alphaxiv.org/?sort=Likes&interval=7+Days"
req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urlopen(req, timeout=30).read().decode("utf-8", errors="replace")

with open("alphaxiv.html", "w", encoding="utf-8") as f:
    f.write(html)

print(html)