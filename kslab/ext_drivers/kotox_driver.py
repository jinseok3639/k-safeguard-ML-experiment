"""KOTOX 규칙 코드 드라이버. `kslab/external_gen.py`가 KOTOX 전용 venv의 파이썬으로 실행한다.

사용: <kotox venv python> -X utf8 -I kotox_driver.py <kotox-src 경로>  (표준입력 JSONL -> 표준출력 JSONL)

입력 한 줄: {"text": ..., "op": "rule:5-2" | "compose:3", "seed": int}
출력 한 줄: {"out": ..., "rules": [...]}  (규칙이 안 걸리면 out은 원문, rules는 빈 목록)
          compose는 "steps"(규칙마다 끝난 뒤의 문장)도 준다.

KOTOX `augmentation.py`의 `Augmentation`을 그대로 쓴다. GPT-4.1을 부르는 8-1·8-3은 빼고, G2P 저장소가 필요한
1-5는 그 저장소(KoG2Padvanced, 라이선스 표시 없음 — 내부 평가에만 쓴다)가 없으면 뺀다. openai·dotenv·pandas·tqdm은 결정론적 규칙에 쓰이지 않아 빈 모듈로 대신한다.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import types

LLM_RULES = ("8-1", "8-3")
APPLY_RATIO = 0.4   # KOTOX main()의 기본값


def _stub_modules(root: str) -> bool:
    """결정론적 규칙에 필요 없는 의존성을 빈 모듈로 막는다. G2P가 있으면 True."""
    openai = types.ModuleType("openai")
    openai.OpenAI = lambda **_: None
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules.update({"openai": openai, "dotenv": dotenv,
                        "pandas": types.ModuleType("pandas"), "tqdm": types.ModuleType("tqdm")})
    sys.modules["tqdm"].tqdm = lambda x, **_: x
    g2p_dir = os.path.join(root, "G2P")
    if os.path.isdir(os.path.join(g2p_dir, "G2P")):
        # KoG2Padvanced 저장소를 kotox-src/G2P 로 받으면 패키지는 그 안의 G2P/ 다
        sys.path.insert(0, g2p_dir)
        return True
    g2p = types.ModuleType("G2P")
    g2p_adv = types.ModuleType("G2P.KoG2Padvanced")

    def missing(_):
        raise RuntimeError("G2P 저장소가 없어 1-5를 쓸 수 없습니다.")

    g2p_adv.KoG2Padvanced = missing
    sys.modules.update({"G2P": g2p, "G2P.KoG2Padvanced": g2p_adv})
    return False


def _load_augmentation(root: str):
    sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("kotox_augmentation", os.path.join(root, "augmentation.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Augmentation


def main() -> None:
    root = os.path.abspath(sys.argv[1])
    has_g2p = _stub_modules(root)
    Augmentation = _load_augmentation(root)
    os.chdir(root)   # 규칙 사전을 ./rules/ 상대 경로로 연다(생성자에서만)
    aug = Augmentation(random.Random(0))
    for code in set(LLM_RULES) | (set() if has_g2p else {"1-5"}):
        aug.MAP.pop(code, None)
    if has_g2p:
        os.chdir(os.path.join(root, "G2P"))   # G2P는 호출할 때마다 ./G2P/Dic/ 을 연다

    for line in sys.stdin:
        item = json.loads(line)
        random.seed(item["seed"])   # 규칙 함수는 전역 random을 쓴다
        aug.rng = random.Random(item["seed"])
        kind, arg = item["op"].split(":")
        text = item["text"]
        if kind == "compose":
            report = aug.augmentation([text, text], int(arg), APPLY_RATIO)
            steps = report["toxic_steps"]
            result = {"out": steps[-1]["obfuscated_text"] if steps else text, "rules": report["obfuscated_rules"],
                      "steps": [s["obfuscated_text"] for s in steps]}
        else:
            result = _single(aug, text, arg)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def _single(aug, text: str, code: str) -> dict:
    """규칙 하나만 건다. `Augmentation.augmentation`이 한 단계에서 하는 일과 같다."""
    if code not in aug.MAP:
        raise ValueError(f"쓸 수 없는 규칙: {code}")
    spans = [{"span": [w], "applied_rule": []} for w in aug._tokenize(text)]
    if code == "10":
        out = aug.MAP[code](spans)
    elif code in aug.SENTENCE:
        out = aug.MAP[code](aug._detokenize(spans))
    else:
        # 첫 단계에서 LAST 규칙은 _is_possible이 막으므로 바로 건다
        after = [aug.MAP[code](s["span"][-1]) for s in spans]
        after = [a if a != s["span"][-1] else None for a, s in zip(after, spans)]
        chosen = aug._select_span(after, APPLY_RATIO, code)
        if not chosen:
            return {"out": text, "rules": []}
        for i in chosen:
            spans[i]["span"].append(after[i])
        out = aug._detokenize(spans)
    return {"out": out, "rules": [code] if out != text else []}


if __name__ == "__main__":
    main()
