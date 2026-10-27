"""통합 난독화 분류표.

KOTOX(arXiv 2510.10961)의 5분류(음운·모양·음차·구조·화용)를 뼈대로 쓰고, KOTOX에 없는 유형(자모 분해,
음절 늘이기, 초성체)을 덧붙인다. 생성 기법은 모두 이 표의 한 칸에 속하고, 생성기·가드레일·LLM·복원기
실험의 결과표는 이 표의 행으로 집계한다.

- `kotox`: KOTOX 데이터의 규칙 코드(`augmentation.py` 기준).
- `konte`: KoNTe(실제 숙소 후기 1,000건, 단독 주석)의 유형 이름과 건수. 한 건수를 여러 유형이 나눠 갖는
  경우가 있어 합계에 쓰지 않는다. 사람형 공격자 비율의 참고값일 뿐이다.
- `scope`: 복원기 범위. "target"은 되돌릴 대상, "boundary"는 생성만 하고 복원 여부는 P04에서 정함,
  "out"은 복원기가 다루지 않음(의미 번역처럼 되돌리는 문제가 아니거나, 초성체처럼 정보가 너무 많이 사라짐).
  가드레일·LLM 실험은 범위와 상관없이 모두 잰다.
"""

from __future__ import annotations

from dataclasses import dataclass

from kslab.obfuscate import TRANSFORMS

SCOPES = ("target", "boundary", "out")
CATEGORIES = ("음운", "모양", "음차", "구조", "화용", "KOTOX 밖")


@dataclass(frozen=True)
class ObfType:
    id: str
    name: str
    category: str
    example: str
    kotox: tuple[str, ...]
    konte: str
    scope: str
    techniques: tuple[str, ...] = ()   # 지금 있는 내부 기법(`obfuscate.TRANSFORMS`)
    planned: tuple[str, ...] = ()      # P01.2에서 만들 내부 기법
    note: str = ""


TYPES: tuple[ObfType, ...] = (
    ObfType("phon_initial", "초성 교체·된소리", "음운", "시스템→씨스템", ("1-1",), "음소 기반 교체 903",
            "target", ("tensify", "tensify_rule", "aspirate")),
    ObfType("phon_vowel", "모음 교체", "음운", "ㅐ↔ㅔ (해줘→헤줘)", ("1-3",), "음소 기반 교체 903",
            "target", ("vowel_confuse",)),
    ObfType("phon_coda", "받침 교체", "음운", "돈→돉", ("1-4",), "음소 기반 교체 903",
            "target", ("coda_neutralize", "cluster_reduce"), note="규칙형(표준발음법)만 있다"),
    ObfType("phon_spelling", "소리 나는 대로 적기", "음운", "할 짓이가→할찌시가", ("1-5",), "표음주의 표기 408",
            "target", ("nasalize", "lateralize", "palatalize", "h_deletion", "liaison_correct")),
    ObfType("phon_insert", "받침·반모음·초성 추가", "음운", "많이→많휘, 거품→궈퓸", ("2-1", "2-2", "2-3"),
            "자모 추가 360", "target", ("jongseong_cram", "semivowel_insert", "initial_insert", "jamo_copy"),
            note="jamo_copy는 yu2024 COPY(쓰레기→쓸레기)"),
    ObfType("phon_liaison", "연음", "음운", "들어봐→드러봐", ("3-1",), "표음주의 표기 408",
            "target", ("liaison",)),
    ObfType("shape_foreign", "닮은 비한글 문자", "모양", "ㄱ→7·ガ, 가→㉮, 야구→OF구", ("6-1",), "형태 기반 교체 56",
            "boundary", ("lookalike_foreign",),
            note="한글→비한글. 되돌릴 답이 하나. 단어 회전(rotation_dict)도 여기"),
    ObfType("shape_hangul", "닮은 한글(야민정음)", "모양", "귀엽다→커엽다, 대↔머", ("5-1", "5-2"), "형태 기반 교체 56",
            "boundary", ("lookalike_hangul",),
            note="한글↔한글. 양쪽이 실제 글자라 정상 문장 오변경 위험"),
    ObfType("translit", "라틴·한자 음차", "음차", "게시판→gㅔ시판", ("8-1", "8-2"), "발음 기반 교체 96",
            "boundary", note="KOTOX 8-1은 GPT-4.1 생성"),
    ObfType("semantic", "의미 번역", "음차", "가지 말고→돈트 고", ("8-3",), "언어 전환 120",
            "out", note="원문으로 되돌리는 문제가 아니라 의미 이해 문제"),
    ObfType("spacing", "띄어쓰기 교란", "구조", "시 스 템", ("10",), "띄어쓰기 삭제 105",
            "target", ("break_spacing",)),
    ObfType("anagram", "글자 순서 섞기", "구조", "프롬트프", ("11",), "글자 순서 교체 17",
            "boundary"),
    ObfType("symbol_insert", "기호·이모티콘 삽입", "화용", "돈을 °♡ 쓰는", ("13-2",), "이모티콘 150, 글자 추가 53",
            "target", ("zwsp_inject", "symbol_insert"),
            note="zwsp_inject는 투명 문자만 넣는다"),
    ObfType("jamo_decompose", "자모 분해", "KOTOX 밖", "쓰레기→ㅆㅡ레기", (), "음절 해체 123",
            "target", ("jamo_decompose",), note="근거: yu2024(NAACL Findings), KoNTe"),
    ObfType("syllable_extend", "음절 늘이기", "KOTOX 밖", "화장실→호와당실", (), "음절 확장 121",
            "boundary", ("syllable_extend",), note="근거: KoNTe뿐(약함)"),
    ObfType("chosung", "초성체", "KOTOX 밖", "안녕하세요→ㅇㄴㅎㅅㅇ", (), "자모 삭제 33 일부",
            "out", ("chosung",),
            note="실제 관찰 근거 약함. 위협 조건(D2)으로 채택. 중성·종성이 사라져 복원이 가장 어려워 복원기 범위에서 "
                 "뺐다. 생성기는 가드레일·LLM 실험용으로 남긴다"),
)

BY_ID = {t.id: t for t in TYPES}


def type_of(technique: str) -> ObfType:
    """기법(지금 있는 것 또는 P01.2 예정)이 속한 유형."""
    for t in TYPES:
        if technique in t.techniques or technique in t.planned:
            return t
    raise KeyError(f"분류표에 없는 기법: {technique}")


def unassigned() -> list[str]:
    """`obfuscate.TRANSFORMS`에 있는데 분류표 어느 칸에도 없는 기법."""
    assigned = {name for t in TYPES for name in t.techniques}
    return sorted(set(TRANSFORMS) - assigned)


def coverage_rows() -> list[dict[str, str]]:
    """유형별 커버리지 표의 행. 외부 생성기 칸은 P01.3에서 채운다."""
    rows = []
    for t in TYPES:
        if t.techniques:
            status = "있음"
        elif t.planned:
            status = "P01.2 예정"
        else:
            status = "없음"
        rows.append({
            "분류": t.category, "유형": t.name, "id": t.id, "KOTOX": ", ".join(t.kotox) or "—",
            "KoNTe": t.konte, "범위": t.scope, "내부 생성기": status,
            "기법": ", ".join(t.techniques + tuple(f"{p}*" for p in t.planned)) or "—",
        })
    return rows
