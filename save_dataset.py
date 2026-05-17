from huggingface_hub import hf_hub_download
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

repo = "deepmind/pg19"
split = "test"
max_docs = 100          # 처음엔 50~200으로 시작 권장
out = "/tmp/pg19_test.jsonl"
asset_root_url = "https://storage.googleapis.com/deepmind-gutenberg/"

index_path = hf_hub_download(
    repo_id=repo,
    repo_type="dataset",
    filename=f"data/{split}_files.txt",
)

with open(index_path, encoding="utf-8") as f:
    rel_files = [x.strip() for x in f if x.strip()]

def download_pg19_text(rel: str) -> str:
    """
    deepmind/pg19의 실제 본문 파일은 HF repo가 아니라
    https://storage.googleapis.com/deepmind-gutenberg/ 아래에 위치한다.
    (pg19.py의 _ASSET_ROOT_URL 참고)
    """
    normalized = rel.lstrip("./").lstrip("/")
    url = asset_root_url + normalized
    req = Request(url, headers={"User-Agent": "pg19-jsonl-builder/1.0"})
    with urlopen(req, timeout=60) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore").strip()

n = 0
skipped = 0
with open(out, "w", encoding="utf-8") as wf:
    for rel in rel_files[:max_docs]:
        try:
            text = download_pg19_text(rel)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[WARN] failed to download {rel}: {exc}")
            skipped += 1
            continue
        if not text:
            skipped += 1
            continue
        wf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        n += 1

print(f"saved={out}, docs={n}, skipped={skipped}")
