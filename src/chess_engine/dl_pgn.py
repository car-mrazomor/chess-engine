import requests
import zipfile 
import io
import os
from selectolax.lexbor import LexborHTMLParser
import time


def _url(path: str = "") -> str:
    return f"https://www.pgnmentor.com/{path}"


def _request(path: str = "files.html", method: str = "GET") -> requests.Response:
    headers: dict = {"User-Agent":
               "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36"}

    return requests.request(method, _url(path), headers=headers)


def parse_html(res: requests.Response) -> list[str]:
    tree = LexborHTMLParser(res.text)
    array: list[str] = []
    for link in tree.css("a"):
        try:
            if link:
                if not link.attributes.get("href").startswith("https") and link.text() == "Download":
                    array.append(link.attributes.get("href"))
        except:
            continue

    return array


def dl_func():
    array_links: list = parse_html(_request())

    print("DL started")
    start = time.perf_counter()

    for i, link in enumerate(array_links):
        os.makedirs("PGNs", exist_ok=True)
        res = _request(link)
        basename = os.path.basename(link)

        if ".pgn" in basename:
            with open(f"PGNs/{basename}", "w") as f:
                f.write(res.text)
        else:
            with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
                zf.extractall("PGNs")

        print(f" {i+1} dl | -> {link} | finished")

    print(f"DL finished too -> {start - time.perf_counter()}")


if __name__ == "__main__":
    dl_func()
