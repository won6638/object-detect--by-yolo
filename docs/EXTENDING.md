# Extending the Project

이 문서는 새로운 VLM 또는 새로운 UI를 붙일 때 어디를 수정해야 하는지 빠르게 설명합니다.

---

# 1. VLM 교체

현재:

```text
Harness -> LlamaCppProvider -> Qwen2.5-VL-3B
```

목표 예:

```text
Harness -> OpenAIProvider -> Commercial VLM
```

새 Provider가 지켜야 할 핵심 Contract:

```python
class NewProvider:
    def __init__(self, cfg, schema):
        ...

    def health(self):
        ...

    def identify(
        self,
        *,
        image_path,
        selection_method,
        yolo_hint,
        quality_flags,
    ):
        return parsed_json, latency_ms, metadata
```

Provider는 `vision_response.schema.json`과 호환되는 JSON을 만들어야 합니다.

현재 Harness는 `LlamaCppProvider`를 직접 import/생성하므로 새 Provider 도입 시 `harness.py`의 Provider 생성 지점을 수정해야 합니다.

장기적으로는 Provider Factory 도입을 권장합니다.

---

# 2. VLM 교체 시 건드리지 말아야 할 영역

가급적 다음은 수정하지 않습니다.

```text
YOLOE
BoT-SORT
Capture folder layout
object_id / capture_id / request_id
queue.jsonl
record.json
Validator
HUD result contract
```

새 Provider는 이 구조 위에서 동작해야 합니다.

---

# 3. Prompt 변경

Prompt는 Provider 계층에 둡니다.

Vision 코드에 특정 VLM Prompt를 작성하지 않습니다.

이유:

```text
Vision = Where / Which
VLM = What
```

역할을 유지하기 위해서입니다.

---

# 4. UI 교체

현재 Desktop Renderer:

```text
OpenCV
```

UI가 필요로 하는 최소 데이터:

```text
object_id
request_id
current target position
llm.status
llm.response
```

즉 UI는 Provider 종류를 알 필요가 없습니다.

---

# 5. VR UI 구현 시 권장 구조

예:

```text
Unity / Quest
│
├─ Selection Controller
├─ Result State Store
├─ Object Anchor Manager
└─ Result Card Renderer
```

Result Card가 보여줄 값:

```text
object_name
category
brand
product_name
model
status
```

Raw Provider Response는 직접 사용하지 않고 Harness의 Validated Response를 사용합니다.

---

# 6. Desktop HUD를 제거하는 위치

현재 파일:

```text
yoloe_desktop_pipeline_v2_2_ui.py
```

관련 함수:

```text
poll_result_cards
result_card_lines
draw_result_cards
draw_info_card
```

VR 전환 시:

- `poll_result_cards`의 역할은 Result State 계층으로 이동
- `draw_result_cards`
- `draw_info_card`

는 Unity/VR Renderer로 대체

하는 방향을 권장합니다.

---

# 7. Transport 교체

현재 Desktop Prototype:

```text
Vision Process
    ↓ files
queue.jsonl / record.json
    ↓
Harness
```

같은 PC에서는 단순하고 Debug가 쉬운 장점이 있습니다.

Quest처럼 다른 Device로 분리되면 File IPC를 그대로 쓰기보다 Network API 계층을 추가할 수 있습니다.

예:

```text
Quest
  ↓ HTTP/WebSocket
PC Vision/VLM Backend
```

이때도 내부 DTO에는 다음 값을 유지하세요.

```text
object_id
request_id
capture_id
status
response
```

---

# 8. Provider 교체 Checklist

새 VLM을 추가한 뒤 확인:

- [ ] `health()` 동작
- [ ] 이미지 입력 가능
- [ ] JSON output 가능
- [ ] Schema 통과
- [ ] Brand evidence rule 통과
- [ ] Product evidence rule 통과
- [ ] Model evidence rule 통과
- [ ] Timeout 처리
- [ ] Provider latency 기록
- [ ] Unknown/uncertain 처리
- [ ] 오래된 Request가 현재 UI에 표시되지 않음

---

# 9. UI 교체 Checklist

새 Renderer를 추가한 뒤 확인:

- [ ] 현재 `object_id`와 결과가 연결됨
- [ ] 최신 `request_id`만 표시
- [ ] pending 표시
- [ ] processing 표시
- [ ] completed 표시
- [ ] uncertain 표시
- [ ] unknown 표시
- [ ] timeout/failed 표시
- [ ] Tracking lost 시 잘못된 객체에 카드가 붙지 않음
- [ ] VLM이 느려도 Rendering loop가 멈추지 않음
