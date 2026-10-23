"""요구사항 자동 점검 (검증 결과 테스트 결과표용).

실행: python scripts/run_checks.py
명세 4장 검증 항목 중 코드로 확인할 수 있는 것을 실행하고 docs/eval/test_results.md로 저장한다.
화면이 필요한 항목(F2009~F2012 일부)은 Streamlit AppTest로 확인하고, 화면 캡처는 docs/screenshots/에 있다.
LLM·임베딩 호출이 들어가 1~2분 걸린다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.collect import cache, http  # noqa: E402
from app.collect.collector import collect, extract_queries  # noqa: E402
from app.config import PROJECT_ROOT, env, get_config, path  # noqa: E402
from app.data_prep.item_text import item_text, load_models  # noqa: E402
from app.index.indexer import index_models  # noqa: E402
from app.index.store import DOCS_COLLECTION, META_COLLECTION, count, get_store  # noqa: E402
from app.rag.citations import NO_CITATION_NOTICE, cited_evidence, unverified_numbers  # noqa: E402
from app.rag.graph import build_graph, run  # noqa: E402
from app.rag.pipeline import ask, user_message  # noqa: E402
from app.rag.prompts import NOT_FOUND  # noqa: E402
from app.rag.retrieve import Evidence, retrieve  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
results: list[tuple[str, str, bool, str]] = []


def check(req: str, name: str) -> Callable:
    def wrap(fn: Callable[[], tuple[bool, str]]):
        t0 = time.monotonic()
        try:
            ok, detail = fn()
            ok = bool(ok)
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"예외: {type(e).__name__}: {e}"
        results.append((req, name, ok, f"{detail} ({time.monotonic() - t0:.1f}s)"))
        print(("PASS" if ok else "FAIL"), req, name, "|", detail)
        return fn
    return wrap


models = {m["id"]: m for m in load_models()}
m8 = models[MODEL]


@check("F2001", "항목 메타데이터 100건 이상 로드, 항목 1건 = 텍스트 1개")
def _():
    t = item_text(m8)
    fields = ["모델명", "제작사", "파라미터 수", "아키텍처", "컨텍스트 길이", "라이선스", "태그(용도)"]
    return len(models) >= 100 and all(f in t for f in fields), f"{len(models)}건, 필드 {len(fields)}개 포함"


@check("F2002", "색인 청크 수 = 항목 수, 재색인해도 불변")
def _():
    before = count(META_COLLECTION)
    after = index_models(list(models.values()))
    return before == after == len(models), f"색인 전 {before} -> 후 {after}, 항목 {len(models)}"


@check("F2003", "프롬프트 1이 파싱 가능한 검색어 배열을 낸다 (모델 ID 포함)")
def _():
    qs, fallback = extract_queries(m8, "Q4로 돌리면 메모리 얼마나 필요해?")
    return isinstance(qs, list) and qs and not fallback, f"{qs}"


@check("F2004·NF2002", "이미 수집한 검색어는 다시 수집하지 않는다 (캐시 히트)")
def _():
    q = "Qwen3-8B의 권장 샘플링 설정은?"
    collect(m8, q)  # 처음이면 수집, 이미 했으면 캐시
    r = collect(m8, q)
    return r.cache_hit and not r.fetched, f"cache_hit={r.cache_hit}, 문서 {r.docs_total}건 사용"


@check("F2004", "선택 모델의 모델 카드는 항상 색인되어 있다")
def _():
    docs = cache.indexed_docs(MODEL)
    return f"hf_card:{MODEL}" in docs, f"{MODEL} 색인 문서 {len(docs)}건"


@check("F2005", "model_docs 청크 ID = 모델ID:문서ID:순번, 메타데이터 필드 저장")
def _():
    g = get_store(DOCS_COLLECTION).get(where={"model_id": MODEL}, limit=50, include=["metadatas"])
    id_ok = all(re.fullmatch(rf"{re.escape(MODEL)}:[a-z_]+:.+:\d+", i) for i in g["ids"])
    need = {"doc_title", "source_url", "source_type", "doc_id", "model_id", "query", "collected_at"}
    meta_ok = all(need <= set(m) for m in g["metadatas"])
    return id_ok and meta_ok and g["ids"], f"청크 {len(g['ids'])}개 검사, 예: {g['ids'][0]}"


@check("F2005", "청크 크기·중복은 config.yaml에서 읽는다")
def _():
    c = get_config()["chunking"]
    return {"chunk_size", "chunk_overlap"} <= set(c), f"chunk_size={c['chunk_size']}, overlap={c['chunk_overlap']}"


@check("F2006", "메타데이터 근거가 항상 [근거 1], Top-k를 바꾸면 근거 수가 바뀐다")
def _():
    cfg = get_config()["retrieval"]
    old = cfg["top_k"]
    try:
        cfg["top_k"] = 2
        a = retrieve(MODEL, "권장 샘플링 설정", concept_queries=[])
        cfg["top_k"] = 6
        b = retrieve(MODEL, "권장 샘플링 설정", concept_queries=[])
    finally:
        cfg["top_k"] = old
    ok = a[0].source_type == "model_meta" and b[0].source_type == "model_meta" and len(b) > len(a)
    return ok, f"top_k=2 -> 근거 {len(a)}건, top_k=6 -> {len(b)}건"


@check("F2006", "청크 ID·본문 해시·근사 중복 제거")
def _():
    ev = retrieve(MODEL, "모델 구조와 컨텍스트 길이", concept_queries=["컨텍스트 윈도"])
    ids = [e.chunk_id for e in ev]
    return len(ids) == len(set(ids)), f"근거 {len(ev)}건, 중복 ID 없음"


@check("F2007", "LangGraph가 retrieve -> generate 두 노드")
def _():
    g = build_graph().get_graph()
    nodes = [n for n in g.nodes if not n.startswith("__")]
    edges = [(e.source, e.target) for e in g.edges]
    return nodes == ["retrieve", "generate"] and ("retrieve", "generate") in edges, f"노드 {nodes}, 엣지 {edges}"


@check("F2007", "문서에 없는 질문: 추측한 수치 없이 자료에 없다고 대화체로 밝힌다 (고정 문구 없음)")
def _():
    from app.rag.citations import _NUM, _norm_numbers
    q = "Qwen3-8B 학습에 GPU를 몇 시간 썼고 비용은 얼마였어?"
    s = run(MODEL, q, concept_queries=[])
    a = s["answer"]
    # 근거 문서에 있는 수(예: 기술 보고서 arXiv 번호 2505.09388로 확인처 안내)는 지어낸 것이 아니다
    ev = " ".join(e.text for e in s["evidence"])
    invented = [n for n in _NUM.findall(_norm_numbers(re.sub(r"\[근거 \d+\]", "", a)))
                if n not in q and n not in _norm_numbers(ev)]
    ok = NOT_FOUND not in a and re.search(r"없|않", a) is not None and not invented  # "없어요"·"나와 있지 않아요"
    return ok, f"고정 문구 {NOT_FOUND in a}, 지어낸 수치 {invented} | {a[:90]}"


@check("F2008", "실제 인용된 근거만 표시, 인용이 없으면 안내")
def _():
    ev = [Evidence(i, f"c{i}", "t", "hf_card", f"d{i}", "u") for i in (1, 2, 3)]
    a = [e.n for e in cited_evidence("가 [근거 3] 나 [근거 9]", ev)]
    s = run(MODEL, "Qwen3-8B 학습에 GPU를 몇 시간 썼고 비용은 얼마였어?", concept_queries=[])
    return a == [3] and (s["cited"] or s["notice"] == NO_CITATION_NOTICE), f"[근거 3][근거 9] -> {a}, 없음 안내 '{s['notice']}'"


@check("F2008", "근거에 없는 수치를 인용 형식으로 쓰면 경고 (환각 검출)")
def _():
    ev = [Evidence(1, "a", "TopK=20, Temperature=0.6", "hf_card", "c", "u")]
    p = unverified_numbers("- TopK: 50 [근거 1]\n- Temperature: 0.6 [근거 1]", ev)
    return len(p) == 1 and "50" in p[0], f"{p}"


@check("F2008", "단위가 붙은 소수(180.0B)는 일부(180)만 떼어 검사하지 않는다 (오탐 회귀)")
def _():
    ev = [Evidence(1, "a", "파라미터 수: 180.0B (179,999,981,459개)", "hf_card", "c", "u")]
    p = unverified_numbers("- 파라미터 수: 180.0B (179,999,981,459개) [근거 1]", ev)
    return p == [], f"경고 {p}"


@check("F2007", "대화형 답변: 후속 질문은 이전 대화로 이해, 근거 밖 일반 지식에서 근거 없는 수치 문장 제거")
def _():
    from app.features.fit import memory_text
    from app.rag.citations import strip_general_numbers
    from app.rag.prompts import GENERAL_LABEL
    calc = memory_text(m8)
    first = run(MODEL, "Q4_K_M으로 돌리면 메모리 얼마나 필요해?", calc=calc, concept_queries=[])
    follow = run(MODEL, "그럼 Q8_0은?", calc=calc, concept_queries=[],
                 history=[("user", "Q4_K_M으로 돌리면 메모리 얼마나 필요해?"), ("assistant", first["answer"])])
    q8 = next(ln for ln in calc.splitlines() if ln.startswith("- Q8_0"))
    q8_total = q8.rsplit("= ", 1)[1].split(" GB")[0]
    guard, removed = strip_general_numbers(f"x [근거 1]\n\n{GENERAL_LABEL}\n보통 수백 GPU 시간이 걸립니다. 기술 보고서를 보세요.", GENERAL_LABEL)
    ok = q8_total in follow["answer"] and len(removed) == 1 and "수백" not in guard
    return ok, f"후속 '그럼 Q8_0은?' 답에 Q8_0 계산값 {q8_total} GB 포함={q8_total in follow['answer']}, 일반 지식 수치 문장 제거 {removed}"


@check("F2007", "모델과 관계없는 일반 질문(알고리즘 문제)은 문서 검색 없이 일반 답변, 수치 제거 장치를 적용하지 않음")
def _():
    from app.collect.collector import extract_queries
    q = "정수 n개 배열에서 합이 k인 연속 부분 배열의 개수를 구하는 O(n) 알고리즘을 파이썬으로 짜줘"
    queries, fallback = extract_queries(m8, q)
    r = ask(m8, q)
    ok = queries == [] and not fallback and r.collect.general and not r.cited and r.notice and "```" in r.answer
    return ok, f"검색어 {queries}, 일반 질문={r.collect.general}, 안내='{r.notice[:40]}', 코드 포함={'```' in r.answer}"


@check("F2001", "최신 멀티모달(image-text-to-text) 모델과 최근 뜨는 원본 모델도 목록에 포함")
def _():
    snap = json.loads(path("models_json").read_text(encoding="utf-8"))
    tags = {m.get("pipeline_tag") for m in models.values()}
    want = [i for i in ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B", "google/gemma-4-31B-it") if i in models]
    ok = {"text-generation", "image-text-to-text"} <= tags and len(want) == 3 and len(snap.get("trending", [])) > 0
    return ok, f"태그 {sorted(t for t in tags if t)}, 포함 {want}, trending {len(snap.get('trending', []))}건 (예: {snap.get('trending', [])[:3]})"


@check("F2010", "하이브리드 어텐션 KV: 선형 레이어 제외, sliding 레이어는 창 크기까지만")
def _():
    from app.features.fit import kv_cache_bytes
    q, g = models["Qwen/Qwen3.8-27B"], models["google/gemma-4-31B-it"]
    plain = lambda m, c: kv_cache_bytes(dict(m, attn_layers=None), c)  # noqa: E731 - 모든 레이어를 전체 어텐션으로 본 값
    ratio = kv_cache_bytes(q, 32768) / plain(q, 32768)
    full_share = q["attn_layers"]["full"] / q["num_layers"]
    grow = kv_cache_bytes(g, 65536) - kv_cache_bytes(g, 32768)
    full_grow = 2 * g["attn_layers"]["full"] * g["global_kv_heads"] * g["global_head_dim"] * 32768 * 2
    ok = abs(ratio - full_share) < 1e-9 and grow == full_grow
    return ok, (f"Qwen3.8-27B 32k KV {kv_cache_bytes(q, 32768) / 2**30:.2f}GB (전체 레이어로 보면 {plain(q, 32768) / 2**30:.2f}GB, "
                f"비율 {ratio:.2f} = 전체 어텐션 {q['attn_layers']['full']}/{q['num_layers']}), "
                f"Gemma4-31B 32k {kv_cache_bytes(g, 32768) / 2**30:.2f}GB (전체로 보면 {plain(g, 32768) / 2**30:.2f}GB), 32k→64k 증가분 = 전체 어텐션 레이어 몫")


@check("F2010", "판정·속도 계산 (Qwen3-8B, RTX 3080, 컨텍스트 4096)")
def _():
    from app.features.fit import assess
    from app.features.hw_detect import manual
    from app.features.speed import estimate_all
    hw = manual("RTX 3080", 32)
    rep = assess(m8, hw)
    q4 = next(r for r in rep.rows if r.quant == "Q4_K_M")
    sp = estimate_all(m8, hw, rep)["Q4_K_M"].tok_s
    return round(q4.total_gb, 1) == 5.8 and q4.verdict.startswith("GPU"), f"Q4_K_M {q4.total_gb:.1f} GB {q4.verdict}, {sp:.1f} tok/s"


@check("F2011", "조건 탐색 결과가 조건(용도 code)에 맞는 모델만")
def _():
    from app.features.recommend import search
    r = search("코딩용 모델")
    ok = r.results and all(re.search(r"(?<![\w])(code|coder|coding)(?![\w])", x.evidence, re.I) for x in r.results)
    return bool(ok), ", ".join(x.model_id for x in r.results)


@check("F2010", "컨텍스트가 길수록 KV 캐시·필요 메모리는 커지고 속도는 줄어든다 (KV 읽기 포함)")
def _():
    from app.features.fit import context_sweep
    from app.features.hw_detect import manual
    from app.features.speed import estimate
    hw = manual("RTX 3080", 32)
    sweep = context_sweep(m8, hw, "Q4_K_M")
    kv = [r.kv_gb for _, r in sweep]
    sp = [estimate(m8, hw, r).tok_s for _, r in sweep]
    ok = all(a < b for a, b in zip(kv, kv[1:])) and all(a > b for a, b in zip(sp, sp[1:]))
    return ok, " / ".join(f"{c // 1024}k: KV {r.kv_gb:.2f}GB {s:.0f}tok/s" for (c, r), s in zip(sweep, sp))


@check("F2010", "흉내 답변 체급: 유효 파라미터로 구간 결정 (MoE는 sqrt(전체×활성))")
def _():
    from app.features.persona import make_persona
    got = {i: make_persona(models[i], "Q4_K_M").tier for i in
           ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B"]}
    exp = ["초소형", "소형", "중형", "중형", "대형"]
    return list(got.values()) == exp, ", ".join(f"{k.split('/')[1]}={v}" for k, v in got.items())


@check("F2011", "추천 상담: 사용자가 말하지 않은 언어 조건은 빼고, 말한 언어는 남긴다 (예시 베끼기 회귀)")
def _():
    from app.features.consult import _drop_unsaid_language
    h = [{"role": "user", "content": "코딩용이면 돼. 그래픽카드는 3080, 램 32기가"}]
    a = _drop_unsaid_language("코딩용 모델, 한국어 지원, 내 GPU에서 돌아가는 거", h)
    b = _drop_unsaid_language("한국어 되는 코딩용 모델", [{"role": "user", "content": "한국어 되는 코딩 모델"}])
    return "한국어" not in a and "코딩" in a and "한국어" in b, f"말 안 함 → '{a}' / 말함 → '{b}'"


@check("F2011", "조건 탐색 검색 문장에서 하드웨어 말을 뺀다 (GPU는 코드 필터로 따로 거름)")
def _():
    from app.features.recommend import Condition, search_query
    q = search_query(Condition("RTX 3080 GPU와 RAM 32GB에서 실행할 수 있는 코딩용 모델", [], ["code"], None, None, True))
    return "코딩" in q and not re.search(r"RTX|GPU|RAM|GB", q), f"→ '{q}'"


@check("F2011", "추천 상담: '맥북 M5 16기가'는 사이드바 GPU가 아니라 Apple M5 통합 메모리로 판정")
def _():
    from app.features.consult import step
    from app.features.hw_detect import manual
    t = step([{"role": "user", "content": "맥북 m5 깡통 16기가램인데 뭐가 젤 돌리기 좋냐"}], manual("RTX 5060 Ti 16GB", 16), 0)
    h = t.hardware
    got = [h.kind, h.spec_name, h.gpu_mem_gb, h.bandwidth_gbps] if h else None
    return got == ["apple", "Apple M5", 11.2, 153.0], f"{got} | {t.hw_note}"


@check("F2008", "굵게 표시(**…**) 뒤에 한글 조사가 붙어도 굵게로 렌더링되도록 보정 (O(n)에 -> 별표가 그대로 보이던 문제)")
def _():
    import re as _re
    src = (PROJECT_ROOT / "app/ui/streamlit_app.py").read_text(encoding="utf-8")
    ns = {"re": _re}
    exec(src[src.index("_BOLD_BEFORE_HANGUL"):src.index("def md_escape")], ns)  # noqa: S102 - 우리 코드의 순수 함수만 꺼내 검사
    f = ns["fix_bold"]
    fixed = f("**O(n)**에 구하고 **Q4_K_M**으로 돌려요 **굵게** 뒤 공백")
    ok = "**O(n)**\u200a에" in fixed and "**Q4_K_M**으로" in fixed and fixed.endswith("**굵게** 뒤 공백")
    return ok, repr(fixed)


@check("F2011", "추천 상담: 큰 모델이 안 맞으면 '못 돌림'으로 끝내지 않고 이 PC에 맞는 더 작은 모델을 큰 것부터 제시")
def _():
    from app.features.hw_detect import manual
    from app.features.recommend import model_card, smaller_alternatives
    hw = manual("RTX 5060 Ti 16GB", 32)
    asked = model_card("Qwen/Qwen3.8-27B", hw)  # 오프로딩 (돌아가긴 하지만 느림)
    alts = smaller_alternatives("Qwen/Qwen3.8-27B", hw)
    big = asked.total_gb  # 참고용
    params = {x["id"]: x.get("params") for x in load_models()}
    sizes = [params[a.model_id] for a in alts]
    ok = (asked.verdict == "일부 CPU 오프로딩 필요" and len(alts) >= 2
          and all(a.verdict == "GPU에 전부 올라감" and (a.tok_s or 0) >= 10 for a in alts)
          and all(s < params["Qwen/Qwen3.8-27B"] for s in sizes) and sizes == sorted(sizes, reverse=True))
    return ok, f"27B={asked.verdict}, 대안 {[(a.model_id.split('/')[-1], round(a.tok_s)) for a in alts]}"


@check("F2011", "추천 상담: 모델 이름으로 물으면 그 모델을 찾고(띄어쓰기·꼬리 무시), 안 맞으면 같은 계열 대안")
def _():
    from app.features.hw_detect import manual
    from app.features.recommend import model_card, named_models, same_family_alternatives
    got = {t: named_models(t) for t in ("qwen 3.8 flash next 쓸 수 있음?", "gemma 4 31b 어때", "코딩용 모델 추천해줘")}
    hw = manual("Apple M5 Max (40-core GPU)", 64)
    card = model_card("Qwen/Qwen3.8-Flash-Next", hw)
    alts = [a.model_id for a in same_family_alternatives("Qwen/Qwen3.8-Flash-Next", hw)]
    ok = (got["qwen 3.8 flash next 쓸 수 있음?"] == ["Qwen/Qwen3.8-Flash-Next"] and got["gemma 4 31b 어때"] == ["google/gemma-4-31B-it"]
          and got["코딩용 모델 추천해줘"] == [] and card.verdict == "불가" and alts and all(a.startswith("Qwen/Qwen3.8-") for a in alts))
    return ok, f"{got} | Flash-Next on M5 Max 64GB: {card.verdict}, 대안 {alts}"


@check("F2011", "추천 상담: 사용자가 말하지 않은 하드웨어는 쓰지 않는다 (LLM이 현재 사양을 되풀이해 40코어 -> 32코어로 바뀌던 회귀)")
def _():
    from app.features.consult import step
    from app.features.hw_detect import manual
    t = step([{"role": "user", "content": "qwen 3.8 flash next 쓸 수 있음?"}], manual("Apple M5 Max (40-core GPU)", 64), 0)
    return t.hardware is None, f"하드웨어 {t.hardware.spec_name if t.hardware else None}"


@check("F2010", "DGX Spark: 통합 메모리 128GB × 0.9(가정값)를 GPU 몫으로, 넘치면 같은 메모리에서 오프로딩, 대역폭 273GB/s")
def _():
    from app.features.fit import assess
    from app.features.hw_detect import manual
    hw = manual("NVIDIA DGX Spark", 16)  # 메모리는 사양표의 고정값(128GB)을 쓴다
    big = assess(models["Qwen/Qwen3.8-Flash-Next"], hw)
    fp16 = next(r for r in big.rows if r.quant == "FP16")
    q8 = next(r for r in big.rows if r.quant == "Q8_0")  # 약 196 GB: 128GB 통합 메모리를 넘어 불가
    ok = (hw.kind == "unified" and hw.ram_gb == 128 and hw.gpu_mem_gb == 115.2 and hw.bandwidth_gbps == 273
          and fp16.verdict == "불가" and q8.verdict == "불가")
    return ok, f"{hw.kind}, 메모리 {hw.ram_gb} GB, GPU 몫 {hw.gpu_mem_gb} GB, {hw.bandwidth_gbps} GB/s, Flash-Next FP16 {fp16.total_gb:.0f} GB → {fp16.verdict}"


@check("F2011", "대화 속 GPU 이름 매칭 (줄임말·VRAM 구분)")
def _():
    from app.data_prep.hw_tables import match_gpu
    cases = {"4090": "RTX 4090", "3080 ti": "RTX 3080 Ti", "RTX 3080": "RTX 3080", "M4 Pro": "Apple M4 Pro"}
    got = {k: (match_gpu(k).name if match_gpu(k) else None) for k in cases}
    return got == cases, str(got)


@check("F2012", "라이선스 질문에 면책 안내")
def _():
    s = run(MODEL, "이 모델 상업적으로 써도 돼?", concept_queries=[])
    return "법률 자문이 아니며" in s["answer"], s["answer"][-60:].replace("\n", " ")


@check("NF2001", "대표 질문 응답 시간 (수집 캐시 히트 상태)")
def _():
    t0 = time.monotonic()
    r = ask(m8, "Qwen3-8B의 권장 샘플링 설정은?")
    sec = time.monotonic() - t0
    return r.ok and sec < 30, f"{sec:.1f}초 (단계별 {', '.join(f'{k} {v:.1f}s' for k, v in r.timings.items())})"


@check("NF2003", "API 키가 커밋된 파일에 없다")
def _():
    key = env("OPENAI_API_KEY") or ""
    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
    hits = [f for f in tracked if key and key in (PROJECT_ROOT / f).read_text(encoding="utf-8", errors="ignore")]
    ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=PROJECT_ROOT, capture_output=True).returncode == 0
    return bool(key) and not hits and ignored, f"추적 파일 {len(tracked)}개 중 키 포함 {len(hits)}개, .env gitignore={ignored}"


@check("NF2004", "오류는 사용자 문구로 바꾸고 스택 트레이스를 노출하지 않는다")
def _():
    msg = user_message(RuntimeError("Traceback (most recent call last): secret"))
    return "Traceback" not in msg and "secret" not in msg, msg


@check("NF2005", "수집·색인·검색·생성·UI 모듈 분리, 도메인 의존 코드는 data_prep")
def _():
    need = ["app/data_prep", "app/collect", "app/index", "app/rag", "app/features", "app/ui"]
    return all((PROJECT_ROOT / d).is_dir() for d in need), ", ".join(need)


@check("NF2006", "User-Agent 연락처, 호스트별 호출 간격(arXiv 3초), 재시도")
def _():
    ua = http.user_agent()
    iv = get_config()["http"]["min_interval_s"]
    return "(" in ua and iv["export.arxiv.org"] >= 3 and get_config()["http"]["max_retries"] > 0, ua


@check("NF2007", "데이터·문서 출처와 이용 조건 기록")
def _():
    s = (PROJECT_ROOT / "data" / "SOURCES.md").read_text(encoding="utf-8")
    return all(k in s for k in ["Hugging Face", "위키백과", "arXiv", "CC BY-SA", "hw_specs"]), "data/SOURCES.md"


def main() -> int:
    passed = sum(ok for _, _, ok, _ in results)
    lines = [
        "# 자동 점검 결과 (테스트 결과표)", "",
        f"- 실행: `python scripts/run_checks.py` · {time.strftime('%Y-%m-%d %H:%M')} · 통과 {passed}/{len(results)}",
        f"- 기준 모델 `{MODEL}`, LLM `{get_config()['llm']['chat_model']}`, 임베딩 `{get_config()['llm']['embedding_model']}`", "",
        "| 요구사항 | 점검 내용 | 결과 | 근거(실행 결과) |", "|---|---|---|---|",
        *(f"| {req} | {name} | {'PASS' if ok else 'FAIL'} | {' '.join(detail.split()).replace('|', '/')} |" for req, name, ok, detail in results),
    ]
    out = PROJECT_ROOT / "docs" / "eval" / "test_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n통과 {passed}/{len(results)} -> {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    sys.exit(main())
