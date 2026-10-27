"""외부 난독화 생성기 래퍼 — 평가셋 2층(로드맵 D4). **학습에는 쓰지 않는다.**

외부 저장소는 `data/ext/<이름>`에 받아 두고, 의존성 버전이 서로 달라 저장소마다 `.venv`를 따로 만든다.
받은 폴더 안의 코드를 직접 실행하지 않고, 우리 쪽 드라이버(`kslab/ext_drivers/`)를 그 venv의 파이썬으로
돌려 경로만 넘긴다. 문장은 표준입력 JSONL로 한 번에 보낸다.

    from kslab import external_gen as X
    X.run("kotka", [X.Item("명작", "yamin", seed=0)])   # -> [{"out": "띵작"}]
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data" / "ext"
DRIVERS = Path(__file__).resolve().parent / "ext_drivers"
# KOTOX 1-5의 G2P가 쓰는 형태소 분석기(Kkma)는 Java가 필요하다. 시스템 Java(JAVA_HOME)가 없으면
# data/ext/jdk 에 풀어 둔 Temurin JDK를 이 실행에만 쓴다.
PORTABLE_JDK = EXT_DIR / "jdk"


@dataclass(frozen=True)
class Generator:
    name: str
    repo: str          # data/ext 아래 폴더
    driver: str
    source: str        # 출처 표기
    license: str


GENERATORS = {
    "kotox": Generator("kotox", "kotox-src", "kotox_driver.py",
                       "https://github.com/leeyejin1231/KOTOX", "MIT"),
    "kotka": Generator("kotka", "kotka", "kotka_driver.py", "https://github.com/Astro36/kotka", "MIT"),
    "jamo_typo": Generator("jamo_typo", "korean-jamo-typo", "jamo_typo_driver.py",
                           "https://github.com/SJLee0311/korean-jamo-typo", "MIT"),
}

# 외부 생성기의 기법 -> 통합 분류표 유형(`kslab/taxonomy.py`). P01.3 커버리지 표의 근거다.
# KOTOX 코드는 `augmentation.py` MAP의 주석 기준. 8-1·8-3(GPT-4.1)은 쓰지 않는다.
TYPE_OF = {
    ("kotox", "rule:1-1"): "phon_initial", ("kotox", "rule:1-3"): "phon_vowel",
    ("kotox", "rule:1-4"): "phon_coda", ("kotox", "rule:1-5"): "phon_spelling",
    ("kotox", "rule:2-1"): "phon_insert", ("kotox", "rule:2-2"): "phon_insert",
    ("kotox", "rule:2-3"): "phon_insert", ("kotox", "rule:3-1"): "phon_liaison",
    ("kotox", "rule:5-1"): "shape_hangul", ("kotox", "rule:5-2"): "shape_foreign",
    ("kotox", "rule:6-1"): "shape_foreign", ("kotox", "rule:8-2"): "translit",
    ("kotox", "rule:10"): "spacing", ("kotox", "rule:11"): "anagram",
    ("kotox", "rule:13-2"): "symbol_insert",
    ("kotka", "squaredog"): "phon_initial", ("kotka", "yamin"): "shape_hangul",
    ("kotka", "shred:0.3"): "jamo_decompose",
}
# 의도적 난독화가 아닌 대조군(키보드 오타). 커버리지 계산에서 뺀다.
CONTROL = {("jamo_typo", f"T{i}:2") for i in range(1, 6)}


@dataclass(frozen=True)
class Item:
    text: str
    op: str
    seed: int = 0


def python_of(gen: Generator) -> Path:
    exe = EXT_DIR / gen.repo / ".venv" / "Scripts" / "python.exe"
    if not exe.exists():
        exe = EXT_DIR / gen.repo / ".venv" / "bin" / "python"
    if not exe.exists():
        raise FileNotFoundError(f"{gen.name}: {EXT_DIR / gen.repo}/.venv 가 없습니다.")
    return exe


def run(name: str, items: list[Item], timeout: float = 600) -> list[dict]:
    """외부 생성기 하나에 문장 묶음을 보내고 결과를 같은 순서로 받는다."""
    gen = GENERATORS[name]
    payload = "".join(json.dumps({"text": i.text, "op": i.op, "seed": i.seed}, ensure_ascii=False) + "\n"
                      for i in items)
    env = dict(os.environ)
    if not env.get("JAVA_HOME") and PORTABLE_JDK.exists():
        env["JAVA_HOME"] = str(PORTABLE_JDK)
        env["PATH"] = str(PORTABLE_JDK / "bin") + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [str(python_of(gen)), "-X", "utf8", "-I", str(DRIVERS / gen.driver), str(EXT_DIR / gen.repo)],
        input=payload, capture_output=True, text=True, encoding="utf-8", timeout=timeout, cwd=ROOT, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{name} 드라이버 실패:\n{proc.stderr[-2000:]}")
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    if len(lines) != len(items):
        raise RuntimeError(f"{name}: 입력 {len(items)}줄, 출력 {len(lines)}줄\n{proc.stdout[-1000:]}")
    return lines
