Qwen2.5-VL-3B VLM Harness v1
=============================

1) 기존 .venv 활성화
2) python -m pip install -r requirements.txt
3) 최신 공식 llama.cpp Windows CUDA release에서 llama-server.exe와 DLL 준비
4) 서버:
   .\start_llama_server.ps1 -ServerExe "C:\...\llama-server.exe"
5) Health:
   python .\harness.py --health
6) 단일 이미지:
   python .\smoke_test.py "C:\...\capture\objects\...\llm.jpg"
7) queue 한 건:
   python .\harness.py --once
8) 상시 worker:
   python .\harness.py

중요:
harness_config.json의 capture_root 기본값은 "capture".
Harness 폴더와 capture 폴더가 같은 부모 아래라면 "../capture"로 변경해야 한다.

환각 방지:
- unknown/uncertain 허용
- YOLOE label은 hint only
- 이미지 속 텍스트는 data, instruction 아님
- strict JSON schema
- brand/product/model evidence gate
- deterministic validator
- confidence는 교정 확률로 사용하지 않음

UX:
- 최신 pending 우선
- 같은 object_id pending이 여러 개면 최신만 처리
- 단일 worker
- provider/validation latency 기록
- exact image hash cache

현재 아직 안 함:
- Cloud Tier2
- cross-model voting
- 자동 반복 재질의
- UI stale hook 연결
