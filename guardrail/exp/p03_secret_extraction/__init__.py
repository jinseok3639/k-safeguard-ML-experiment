"""p03-secret-extraction — Tensor Trust 비밀값 추출 공격을 가드레일과 LLM에 그대로 태운다.

실행 순서와 데이터 준비는 README.md. 데이터는 `--data`로 고른다(en 원문, ko 한국어 번역본).

스크립트 역할과 실행 순서
  tensortrust.py     데이터 읽기·제외 규칙·입력 메시지·판정(논문 코드를 옮긴 것). 다른 스크립트가 같이 쓴다
  run_guardrail.py   가드레일 하나를 채점한다(로컬). 공격문과 진짜 비밀번호를 각각 넣는다
  generate_llm.py    LLM 응답을 생성한다(Colab, vLLM). 노트·MLflow를 거치지 않고 응답 파일만 쓴다
  run_llm.py         생성 파일을 판정·집계해 run으로 기록한다(로컬). 노트 게이트는 여기서 걸린다
  guarded.py         가드레일 점수와 LLM 판정을 id로 합쳐 가드레일 앞단 유출률을 낸다

    python -m guardrail.exp.p03_secret_extraction.run_guardrail --data en --run-id p03.3-kanana --guardrail kanana
    python -m guardrail.exp.p03_secret_extraction.generate_llm --data en --model LGAI-EXAONE/EXAONE-4.0-32B-AWQ \\
        --revision 1ce84bc --out <Drive>/p03.6-exaone4-32b-awq.gen.jsonl
    python -m guardrail.exp.p03_secret_extraction.run_llm --data en --run-id p03.6-exaone4-32b-awq \\
        --generations <받아 온 .gen.jsonl>
    python -m guardrail.exp.p03_secret_extraction.guarded --guard p03.3-kanana p03.4-horizon-base \\
        --llm p03.6-exaone4-32b-awq p03.7-kanana15-8b
"""
