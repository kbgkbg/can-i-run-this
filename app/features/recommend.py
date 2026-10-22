"""조건 기반 모델 탐색 (F2011).

1. 조건 해석 (LLM, JSON): 검색 문장, 언어, 용도, 파라미터 범위, "내 GPU에 올라가야 함" 여부
2. model_meta 후보 검색: 검색 문장으로 의미 검색 + 언어 조건이 있으면 본문 필터("한국어(ko)" 포함) 검색을 합친다.
   임베딩만으로는 "한국어 지원" 같은 조건이 잘 걸리지 않아 (EXAONE이 30위 밖) 본문 필터를 함께 쓴다.
3. 딱 떨어지는 조건은 코드로 거른다: 언어(메타데이터 표기), 파라미터 범위, GPU 적재 가능(하드웨어 판정)
4. 용도처럼 해석이 필요한 조건은 LLM이 고르되, 메타데이터에서 그대로 인용한 근거가 실제로 있는 것만 남긴다.
5. 남은 모델에 하드웨어 판정 결과를 붙여 카드로 돌려준다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import get_config
from app.data_prep.item_text import LANG_NAMES, item_text, load_models
from app.features.fit import GPU_FULL, FitReport, assess
from app.features.hw_detect import Hardware
from app.features.speed import estimate_all
from app.index.store import META_COLLECTION, get_store
from app.log import get_logger
from app.rag.llm import invoke_json
from app.rag.prompts import candidate_model_select_messages, condition_parse_messages

log = get_logger("recommend")

# 용도 범주 -> 메타데이터(모델명·태그)에서 그 용도를 뒷받침하는 단어. 근거 줄에 이 중 하나가 있어야 통과한다.
# LLM이 인용은 그대로 했지만 용도와 무관한 줄(예: "태그: conversational")을 근거로 대는 일을 막는다.
USE_KEYWORDS = {
    "code": ("code", "coder", "coding", "programming"),
    "math": ("math",),
    "reasoning": ("reasoning", "think", "thinking", "r1"),
    "chat": ("chat", "conversational", "instruct"),
    "vision": ("vision", "image-text-to-text", "vl", "visual"),
    "tool-use": ("tool-call", "tool-use", "function-calling", "agent"),
}
# 단어 경계로만 일치시킨다: "qwen-coder"의 coder는 맞고, "custom_code"(원격 코드 실행 허용 표시)의 code는 아니다
_USE_RE = {u: re.compile(r"(?<![\w])(" + "|".join(map(re.escape, ks)) + r")(?![\w])", re.IGNORECASE)
           for u, ks in USE_KEYWORDS.items()}


@dataclass
class Condition:
    query: str
    languages: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)
    min_params_b: float | None = None
    max_params_b: float | None = None
    must_fit_gpu: bool = False


@dataclass
class Recommendation:
    model_id: str
    reason: str
    evidence: str                      # 메타데이터에서 인용한 근거 줄
    url: str
    verdict: str | None = None         # 하드웨어 판정 (가장 정밀하게 돌릴 수 있는 양자화 기준)
    quant: str | None = None
    total_gb: float | None = None
    tok_s: float | None = None


@dataclass
class SearchResult:
    condition: Condition | None
    results: list[Recommendation] = field(default_factory=list)
    candidates: int = 0
    excluded: dict[str, int] = field(default_factory=dict)   # 코드 필터 단계별 제외 수
    rejected_evidence: int = 0                                # 인용 근거가 메타데이터에 없어 버린 수
    notes: list[str] = field(default_factory=list)


def parse_condition(text: str) -> Condition:
    def validate(v: Any) -> Condition:
        if not isinstance(v, dict) or not isinstance(v.get("query"), str):
            raise ValueError("query 없음")
        num = lambda x: float(x) if isinstance(x, (int, float)) else None  # noqa: E731
        return Condition(
            query=v["query"] or text,
            languages=[str(x).lower() for x in v.get("languages") or [] if str(x).strip()],
            uses=[str(x) for x in v.get("uses") or []],
            min_params_b=num(v.get("min_params_b")),
            max_params_b=num(v.get("max_params_b")),
            must_fit_gpu=bool(v.get("must_fit_gpu")),
        )

    return invoke_json(condition_parse_messages(text), validate, "조건해석") or Condition(query=text)


# 검색 문장에서 뺄 하드웨어 표현. GPU에 올라가는지는 must_fit_gpu로 코드에서 따로 거른다.
# 빼지 않으면 "RTX 3080 GPU와 RAM 32GB에서 실행할 수 있는 코딩용 모델"이 GPU·메모리 단어로 검색돼
# 코딩 모델이 후보 30건에서 통째로 빠졌다 (gpt-6-luna가 하드웨어를 query에 넣는 일이 잦았다)
_HW_WORDS = re.compile(r"(?:RTX|GTX|RX|M\d)\s*\w*|\d+\s*(?:GB|기가)|VRAM|RAM|램|GPU|그래픽\s*카드|메모리|내\s*PC|"
                       r"(?:에서|으로)?\s*(?:실행|돌아가|돌릴|구동)\w*", re.IGNORECASE)


def search_query(cond: Condition) -> str:
    q = re.sub(r"\s{2,}", " ", _HW_WORDS.sub(" ", cond.query)).strip(" ,와과및에서")
    return q if len(q) >= 2 else cond.query


def _search_ids(cond: Condition, k: int) -> list[str]:
    store = get_store(META_COLLECTION)
    query = search_query(cond)
    ids = [d.metadata["model_id"] for d in store.similarity_search(query, k=k)]
    for lang in cond.languages:
        tag = f"{LANG_NAMES[lang]}({lang})" if lang in LANG_NAMES else lang
        ids += [d.metadata["model_id"] for d in store.similarity_search(query, k=k, where_document={"$contains": tag})]
    return list(dict.fromkeys(ids))


def _best_fit(model: dict[str, Any], hw: Hardware) -> tuple[FitReport, Any]:
    report = assess(model, hw)
    return report, report.best({q: s.tok_s for q, s in estimate_all(model, hw, report).items()})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _evidence_ok(evidence: str, text: str, uses: list[str]) -> bool:
    """인용 근거가 (1) 메타데이터에 그대로 있고 (2) 아는 용도 범주라면 그 범주의 단어를 담고 있어야 한다.

    근거 줄 대신 메타데이터 전체(모델명 포함)에서 단어를 찾아도 되지만, 화면에 보여 줄 근거가 조건을
    직접 뒷받침해야 하므로 근거 줄 기준으로 본다. 모델명 줄("모델명: Qwen2.5-Coder-7B")도 근거가 될 수 있다.
    """
    if not evidence or evidence not in _norm(text):
        return False
    return all(_USE_RE[u].search(evidence) for u in uses if u in _USE_RE)


def search(text: str, hw: Hardware | None = None) -> SearchResult:
    cfg = get_config()["recommend"]
    cond = parse_condition(text)
    result = SearchResult(condition=cond)
    models = {m["id"]: m for m in load_models()}

    ids = [i for i in _search_ids(cond, cfg["candidate_k"]) if i in models]
    result.candidates = len(ids)

    def drop(reason: str) -> None:
        result.excluded[reason] = result.excluded.get(reason, 0) + 1

    kept: list[dict[str, Any]] = []
    fits: dict[str, Any] = {}
    for i in ids:
        m = models[i]
        params_b = (m.get("params") or 0) / 1e9
        if cond.languages and not set(cond.languages) <= set(m.get("languages") or []):
            drop("언어 표기 없음")
            continue
        if cond.max_params_b is not None and params_b > cond.max_params_b:
            drop("파라미터 상한 초과")
            continue
        if cond.min_params_b is not None and params_b < cond.min_params_b:
            drop("파라미터 하한 미만")
            continue
        if hw is not None:
            report, best = _best_fit(m, hw)
            fits[i] = (report, best)
            if cond.must_fit_gpu and (best is None or best.verdict != GPU_FULL):
                drop("내 GPU에 안 올라감")
                continue
        kept.append(m)
    if cond.must_fit_gpu and hw is None:
        result.notes.append("하드웨어 정보가 없어 'GPU에 올라가야 함' 조건은 적용하지 않았습니다. 사이드바에서 하드웨어를 감지하세요.")
    if cond.languages:
        result.notes.append("언어 조건은 모델 카드 메타데이터에 언어가 표기된 모델만 확인할 수 있습니다 "
                            "(표기가 없는 모델은 실제로 지원하더라도 제외됩니다).")
    if not kept:
        return result

    texts = {m["id"]: item_text(m) for m in kept}
    if cond.uses:
        # 용도 조건은 LLM이 메타데이터를 읽고 판단한다
        block = "\n\n".join(f"[{i}]\n{t}" for i, t in texts.items())

        def validate(v: Any) -> list[dict[str, str]]:
            if not isinstance(v, dict) or not isinstance(v.get("selected"), list):
                raise ValueError("selected 없음")
            return [s for s in v["selected"] if isinstance(s, dict) and s.get("id") in texts]

        picked = invoke_json(candidate_model_select_messages(text, cond.uses, block, cfg["max_results"]), validate, "모델선별") or []
    else:
        picked = [{"id": m["id"], "reason": "언어·크기·하드웨어 조건을 만족합니다.", "evidence": ""} for m in kept]

    for s in picked[:cfg["max_results"]]:
        evidence = _norm(str(s.get("evidence", "")))
        # 인용 근거가 실제 메타데이터 텍스트에 없으면 LLM이 지어낸 것으로 보고 버린다
        if cond.uses and not _evidence_ok(evidence, texts[s["id"]], cond.uses):
            result.rejected_evidence += 1
            continue
        m = models[s["id"]]
        rec = Recommendation(s["id"], str(s.get("reason", "")), evidence, m["hf_url"])
        if hw is not None:
            report, best = fits[s["id"]]
            if best is not None:
                rec.verdict, rec.quant, rec.total_gb = best.verdict, best.quant, best.total_gb
                rec.tok_s = estimate_all(m, hw, report)[best.quant].tok_s
            else:
                rec.verdict = report.rows[0].verdict if report.rows and report.missing else "불가"
        result.results.append(rec)
    return result
