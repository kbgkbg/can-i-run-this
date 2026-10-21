"""이미 수집한 문서 원문(data/cache/docs)을 오프라인 묶음(data/offline_docs)으로 내보낸다 (F2004).

실행: python scripts/export_offline_docs.py
이후 config.yaml의 collect.offline_mode를 true로 두면 외부 API 없이 이 묶음에서 수집한다.
arXiv 논문 본문은 논문마다 재배포 조건이 달라(arXiv 기본 라이선스는 제3자 재배포를 허용하지 않을 수 있다) 내보내지 않는다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.collect import cache  # noqa: E402
from app.config import path  # noqa: E402


def main() -> int:
    out = path("offline_docs_dir")
    out.mkdir(parents=True, exist_ok=True)
    # 모델에 실제로 색인된 문서만 (선별 오류로 받았다가 색인에서 지운 원문은 캐시에 남아 있을 수 있다)
    indexed = {doc_id for entry in cache.load().values() for doc_id in entry.get("docs", {})}
    docs = [d for d in cache.all_docs() if d.doc_id in indexed and d.source_type != "arxiv_pdf"]
    for doc in docs:
        name = re.sub(r"[^0-9A-Za-z._-]+", "_", doc.doc_id)
        (out / f"{name}.json").write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"{len(docs)}건 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
