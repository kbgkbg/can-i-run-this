"""검색어 -> 후보 수집 -> 선별 -> 본문 수집 -> 색인 (F2003~F2005). 질문 시점에 실행한다.

1. 프롬프트 1로 검색어 추출 (실패 시 1회 재요청, 그래도 실패하면 모델 ID를 기본 검색어로)
2. 캐시에 없는 검색어만 출처별로 후보(제목 + 요약) 수집
3. 프롬프트 2로 후보 선별 (본문을 받기 전에 걸러 외부 호출을 줄인다)
4. 선별 문서 + 선택 모델의 모델 카드(항상, 어떤 제외 규칙보다 우선) 본문 수집
5. model_docs에 색인하고 캐시에 기록
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.collect import cache, http
from app.collect.sources import hf_card, offline, wikipedia
from app.collect.sources.base import Candidate, Document
from app.config import get_config
from app.data_prep.item_text import item_text
from app.index.indexer import index_document
from app.log import get_logger
from app.rag.intent import is_license_question
from app.rag.llm import invoke_json
from app.rag.prompts import candidate_selection_messages, query_extraction_messages

log = get_logger("collector")

StepCallback = Callable[[str, str], None]


@dataclass
class CollectResult:
    queries: list[str] = field(default_factory=list)
    concept_queries: list[str] = field(default_factory=list)    # 위키백과로 라우팅된 개념 검색어 (검색 단계에서도 사용)
    new_queries: list[str] = field(default_factory=list)        # 캐시에 없어 실제로 수집한 검색어
    query_fallback: bool = False                                 # 프롬프트 1 실패로 기본 검색어를 썼는지
    candidates: list[Candidate] = field(default_factory=list)
    filtered: list[str] = field(default_factory=list)            # 코드 필터로 뺀 다른 모델 저장소 (선별 전)
    selected: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    selection_reason: str = ""
    forced: list[str] = field(default_factory=list)              # 선별과 무관하게 강제로 넣은 문서 (모델 카드, 라이선스 원문)
    fetched: list[Document] = field(default_factory=list)        # 이번에 색인한 문서
    chunks_added: int = 0
    cache_hit: bool = False                                      # 모든 검색어가 이미 수집돼 있었는지
    docs_total: int = 0                                          # 이 모델에 색인된 문서 수
    errors: list[str] = field(default_factory=list)              # 사용자에게 보여도 되는 안내 문구


def _noop(stage: str, message: str) -> None:
    pass


def extract_queries(model: dict[str, Any], question: str) -> tuple[list[str], bool]:
    """프롬프트 1. (검색어 목록, 기본값 대체 여부)"""
    max_q = get_config()["collect"]["max_queries"]

    def validate(value: Any) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(q, str) for q in value):
            raise ValueError("문자열 배열이 아님")
        queries = [q.strip() for q in value if q.strip()]
        if not queries:
            raise ValueError("빈 배열")
        return queries[:max_q]

    queries = invoke_json(query_extraction_messages(item_text(model), question, max_q), validate, "프롬프트1")
    if queries is None:
        return [model["id"]], True
    # 대소문자·공백만 다른 중복 제거
    seen, unique = set(), []
    for q in queries:
        if cache.normalize_query(q) not in seen:
            seen.add(cache.normalize_query(q))
            unique.append(q)
    return unique, False


REPO_QUERY = re.compile(r"/|\b(gguf|awq|gptq|exl2|mlx|fp8)\b", re.IGNORECASE)


def route(query: str, model: dict[str, Any]) -> str:
    """검색어를 보낼 출처를 정한다: 저장소형(모델 ID, 모델명, 양자화 형식) -> hf, 그 외 개념어 -> wiki.

    모든 검색어를 두 출처에 다 보내면 "Qwen/Qwen3-8B"로 위키백과를, "양자화"로 HF 저장소 이름을 찾는 등
    쓸모없는 후보와 외부 호출만 늘어서 나눴다.
    """
    q = query.lower()
    if REPO_QUERY.search(query) or model["name"].lower() in q or model["id"].lower() in q:
        return "hf"
    return "wiki"


# 같은 모델의 양자화·배포 변형으로 인정하는 접미 토큰 (저장소 이름 = 모델명 + 이 토큰들)
_VARIANT_TOKEN = re.compile(
    r"^(gguf|ggml|awq|gptq|exl2|mlx|onnx|fp8|fp16|bf16|f16|f32|int4|int8|bnb|4bit|8bit|quantized|"
    r"i?q\d(_[a-z0-9]+)*|\d+bit)$", re.IGNORECASE)


def same_model_repo(candidate_repo: str, model: dict[str, Any]) -> bool:
    """HF 후보가 선택 모델 자체의 변형 저장소인지 (예: Qwen3-8B -> Qwen3-8B-GGUF, Qwen3-8B-Q4_K_M-GGUF).

    프롬프트 2가 쓸 만한 후보가 없을 때 Qwen3-VL-8B나 abliterated 파생처럼 다른 모델을 고르는 일이 있어,
    선별 전에 코드로 거른다. 저장소 이름이 모델명으로 시작하고(앞에 "원제작사_"가 붙는 관례 허용),
    나머지가 양자화·배포 형식 표기뿐이어야 한다.
    """
    repo = candidate_repo.split("/", 1)[-1].lower()
    name = model["name"].lower()
    org_prefix = f"{model['org'].lower()}_"
    if repo.startswith(org_prefix):
        repo = repo[len(org_prefix):]
    if not repo.startswith(name):
        return False
    parts = [p for p in re.split(r"[-.]", repo[len(name):]) if p]
    return all(_VARIANT_TOKEN.match(p) for p in parts)


def gather_candidates(queries: list[str], model: dict[str, Any], result: CollectResult) -> list[Candidate]:
    cfg = get_config()["collect"]
    n = cfg["candidates_per_query"]
    found: dict[str, Candidate] = {}
    for q in queries:
        if cfg["offline_mode"]:
            searches = [("오프라인 문서", lambda q=q: offline.search(q, n))]
        elif route(q, model) == "hf":
            searches = [("Hugging Face", lambda q=q: hf_card.search(q, n))]
        else:
            searches = [("위키백과", lambda q=q: wikipedia.search(q, n, cfg["wikipedia_langs"]))]
        for name, run in searches:
            try:
                for c in run():
                    if c.source_type == "hf_card" and not same_model_repo(c.extra["model_id"], model):
                        result.filtered.append(c.doc_id)
                        continue
                    found.setdefault(c.doc_id, c)
            except http.ExternalAPIError as e:
                log.warning("후보 검색 실패 %s %r: %s", name, q, e)
                msg = f"{name} 검색에 실패해 일부 후보를 건너뛰었습니다."
                if msg not in result.errors:
                    result.errors.append(msg)
    return list(found.values())


def select_candidates(model_id: str, question: str, candidates: list[Candidate]) -> tuple[list[str], str]:
    """프롬프트 2. (선택된 후보 id, 이유)"""
    max_sel = get_config()["collect"]["max_selected_docs"]
    ids = {c.doc_id for c in candidates}
    brief = [{"id": c.doc_id, "source": c.source_type, "title": c.title, "summary": c.summary[:200]} for c in candidates]

    def validate(value: Any) -> tuple[list[str], str]:
        if not isinstance(value, dict) or not isinstance(value.get("selected"), list):
            raise ValueError("selected 배열이 없음")
        chosen = [i for i in value["selected"] if i in ids]  # 목록에 없는 id(환각)는 버린다
        return chosen[:max_sel], str(value.get("reason", ""))

    picked = invoke_json(candidate_selection_messages(model_id, question, brief, max_sel), validate, "프롬프트2")
    if picked is None:
        # 기본값: 검색어마다 첫 후보
        first_per_query: dict[str, str] = {}
        for c in candidates:
            first_per_query.setdefault(c.query, c.doc_id)
        return list(first_per_query.values())[:max_sel], "선별 응답을 해석하지 못해 검색어별 첫 후보를 사용했습니다."
    return picked


_FETCHERS = {"hf_card": hf_card.fetch, "hf_license": hf_card.fetch_license, "wikipedia": wikipedia.fetch}


def _fetch_body(c: Candidate, forced: bool) -> Document | None:
    """본문 수집. 이미 다른 모델 때문에 받아 둔 원문이 있으면 재사용한다."""
    cfg = get_config()["collect"]
    doc = cache.load_doc(c.doc_id)
    if doc is None:
        doc = offline.fetch(c) if cfg["offline_mode"] else _FETCHERS[c.source_type](c)
        if doc is None:
            return None
        cache.save_doc(doc)
    doc.query = c.query
    # 너무 긴 문서는 앞부분만 쓴다. 강제 포함 문서(선택 모델의 카드·라이선스 원문)는 이 규칙에서 제외한다.
    if not forced and len(doc.text) > cfg["max_doc_chars"]:
        doc.text = doc.text[:cfg["max_doc_chars"]]
    return doc


def collect(model: dict[str, Any], question: str, on_step: StepCallback = _noop) -> CollectResult:
    model_id = model["id"]
    result = CollectResult()
    data = cache.load()
    entry = cache.model_entry(data, model_id)
    own_card = hf_card.card_candidate(model_id)

    on_step("검색어 추출", "질문에서 외부 문서 검색어를 뽑는 중")
    result.queries, result.query_fallback = extract_queries(model, question)
    result.concept_queries = [q for q in result.queries if route(q, model) == "wiki"]
    result.new_queries = [q for q in result.queries if cache.normalize_query(q) not in entry["queries"]]
    need_card = own_card.doc_id not in entry["docs"] and not entry.get("no_card")
    # 라이선스 질문이면 저장소의 LICENSE 원문도 강제로 넣는다 (라이선스 이름만으로 조건을 추측하지 않게)
    own_license = hf_card.license_candidate(model_id, model.get("license_link"))
    need_license = (is_license_question(question) and own_license.doc_id not in entry["docs"]
                    and not entry.get("no_license"))

    if not result.new_queries and not need_card and not need_license:
        result.cache_hit = True
        result.docs_total = len(entry["docs"])
        on_step("수집 생략", f"이미 수집된 문서 {result.docs_total}건 사용")
        return result

    on_step("후보 수집", f"검색어 {len(result.new_queries)}개로 후보 문서 검색 중")
    candidates = gather_candidates(result.new_queries, model, result)
    # 모델 카드는 선별 대상이 아니라 강제 포함, 이미 색인한 문서는 다시 고르지 않는다
    result.candidates = [c for c in candidates if c.doc_id != own_card.doc_id and c.doc_id not in entry["docs"]]

    if result.candidates:
        on_step("후보 선별", f"후보 {len(result.candidates)}건 중 색인할 문서 고르는 중")
        result.selected, result.selection_reason = select_candidates(model_id, question, result.candidates)
    result.rejected = [c.doc_id for c in result.candidates if c.doc_id not in result.selected]

    # (후보, 강제 포함 종류) 종류: "card" | "license" | None(선별된 문서)
    to_fetch: list[tuple[Candidate, str | None]] = []
    if need_card:
        to_fetch.append((own_card, "card"))
    if need_license:
        to_fetch.append((own_license, "license"))
    result.forced = [c.doc_id for c, _ in to_fetch]
    by_id = {c.doc_id: c for c in result.candidates}
    to_fetch += [(by_id[i], None) for i in result.selected]

    on_step("본문 수집", f"문서 {len(to_fetch)}건 본문 수집 중")
    offline_mode = get_config()["collect"]["offline_mode"]
    missing_msg = {"card": "이 모델은 모델 카드(README)가 없습니다.",
                   "license": "이 모델 저장소에는 LICENSE 파일이 없어 라이선스 원문을 근거로 쓸 수 없습니다."}
    for c, kind in to_fetch:
        try:
            doc = _fetch_body(c, kind is not None)
        except http.ExternalAPIError as e:
            log.warning("본문 수집 실패 %s: %s", c.doc_id, e)
            gated = "401" in str(e) and model.get("gated")
            result.errors.append(f"'{c.title}'은(는) gated 저장소라 HF_TOKEN 없이는 받을 수 없습니다." if gated
                                 else f"'{c.title}' 본문을 가져오지 못했습니다.")
            continue
        if doc is None:
            if kind and offline_mode:
                # 묶음에 없을 뿐 저장소에는 있을 수 있으므로 기록하지 않는다 (온라인에서 다시 시도)
                result.errors.append(f"오프라인 문서 묶음에 '{c.title}'이(가) 없습니다.")
            elif kind:
                entry[f"no_{kind}"] = True  # 저장소에 파일이 없음: 다음 질문부터 다시 시도하지 않는다
                result.errors.append(missing_msg[kind])
            continue
        result.fetched.append(doc)

    on_step("색인", f"문서 {len(result.fetched)}건 청킹·임베딩 중")
    for doc in result.fetched:
        result.chunks_added += index_document(doc, model_id)
    # 색인까지 끝난 뒤에만 캐시에 기록한다. 중간에 예외가 나면 기록하지 않아 다음 질문에서 다시 수집한다
    # (index_document는 기존 청크를 지우고 다시 넣으므로 재시도해도 중복되지 않는다).
    cache.record(data, model_id, result.new_queries, result.fetched)
    cache.save(data)

    result.docs_total = len(cache.model_entry(data, model_id)["docs"])
    on_step("수집 완료", f"새 문서 {len(result.fetched)}건 색인 (청크 {result.chunks_added}개)")
    return result
