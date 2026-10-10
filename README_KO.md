# YOLO-VLM Vision Pipeline

**한국어** | [English](README.md)

**YOLOE**, **BoT-SORT 추적**, 사용자 객체 선택, 그리고 **Qwen2.5-VL 기반 로컬 VLM 추론**을 결합한 모듈형 실시간 비전 파이프라인입니다.

이 시스템은 실시간 카메라 입력에서 객체를 탐지하고 추적하며, 사용자가 탐지된 객체를 직접 선택하거나 관심 영역(ROI)을 수동으로 지정할 수 있습니다. 선택된 영역의 이미지와 메타데이터는 비동기 방식으로 로컬 Vision-Language Model에 전달되어 객체 식별 및 분석에 사용됩니다.

본 프로젝트는 원래 XR AI 프로토타입의 비전/백엔드 구성요소로 개발되었으며, 현재는 다양한 카메라, VLM 백엔드, XR 인터페이스, 그리고 향후 모션 하드웨어까지 독립적으로 교체하거나 확장할 수 있는 모듈형 비전 시스템으로 발전시키고 있습니다.

---

## 프로젝트 개요

본 파이프라인은 실시간 비전 처리와 VLM 추론을 서로 분리합니다.

```text
Camera
  ↓
YOLOE Detection / Segmentation
  ↓
BoT-SORT Tracking
  ↓
Object Selection / Manual ROI
  ↓
Image + Metadata Capture
  ↓
VLM Request Queue
  ↓
Local Qwen2.5-VL via llama.cpp
  ↓
Schema Validation / Evidence Validation
  ↓
Desktop HUD
```

비전 루프는 VLM의 응답을 기다리지 않습니다.

객체 탐지, 추적, 렌더링, VLM 추론은 서로 분리된 처리 경계를 가지며, VLM 추론이 느리더라도 카메라 및 비전 파이프라인이 멈추지 않도록 설계했습니다.

---

## 주요 기능

- 실시간 카메라 입력
- YOLOE 기반 객체 탐지 및 세그멘테이션
- BoT-SORT 기반 객체 추적
- 마우스 클릭을 통한 객체 선택
- 드래그를 통한 수동 ROI 선택
- 애플리케이션 수준의 지속적인 객체 식별자 관리
- 이미지 및 메타데이터 자동 저장
- JSON 기반 요청 및 결과 기록
- 비동기 VLM 처리
- llama.cpp를 통한 로컬 Qwen2.5-VL 추론
- 구조화된 JSON VLM 응답
- JSON Schema 검증
- 결정론적 증거 검증
- `request_id` 기반 오래된 응답 방지
- SHA-256 기반 동일 이미지 VLM 캐시
- OpenCV 기반 Desktop HUD
- 향후 확장을 고려한 VLM/UI 모듈 경계 설계

---

## 시스템 구조

통합 실행기는 여러 개의 독립 프로세스를 실행합니다.

```text
run_integrated.py
│
├── llama-server.exe
│
├── VLM Harness
│   └── Qwen2.5-VL Provider
│
└── Vision Process
    ├── Camera
    ├── YOLOE
    ├── BoT-SORT
    ├── Selection
    ├── Capture
    └── OpenCV HUD
```

현재 Vision Process와 VLM Harness 사이의 통신은 파일 기반 IPC 방식으로 이루어집니다.

```text
capture/
├── queue.jsonl
├── events.jsonl
└── ...
    ├── original.jpg
    ├── llm.jpg
    └── record.json
```

`record.json`은 다음 정보를 연결하는 핵심 데이터 계약 역할을 합니다.

- 객체 식별자
- 캡처 메타데이터
- Detector 정보
- VLM 요청
- VLM 응답
- 검증 결과
- 지연 시간 정보

보다 자세한 시스템 구조는 다음 문서를 참고하세요.

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/EXTENDING.md`](docs/EXTENDING.md)

---

## 객체 식별자 구조

본 프로젝트는 Tracker의 객체 ID와 애플리케이션 수준의 객체 ID를 구분합니다.

### `track_id`

Tracker가 실행 중에 생성하는 객체 식별자입니다.

프로그램 재실행 또는 Tracking fragmentation이 발생하면 동일한 객체라도 `track_id`가 변경될 수 있습니다.

### `object_id`

사용자가 선택한 객체에 대해 애플리케이션이 생성하는 지속적인 식별자입니다.

```text
object_id
 ├── capture_id
 │    └── request_id
 │
 └── capture_id
      └── request_id
```

이 구조를 사용함으로써 비동기로 반환되는 VLM 결과가 잘못된 객체에 표시되는 문제를 방지합니다.

---

## VLM 파이프라인

현재 VLM 처리 구조는 다음과 같습니다.

```text
Vision Process
    ↓
queue.jsonl
    ↓
VLM Harness
    ↓
LlamaCppProvider
    ↓
llama.cpp
    ↓
Qwen2.5-VL
    ↓
JSON Schema Validator
    ↓
Deterministic Evidence Validator
    ↓
record.json
    ↓
HUD
```

YOLO가 제공하는 객체 이름은 **참고용 힌트**로만 사용하며 정답으로 취급하지 않습니다.

VLM은 실제 캡처된 이미지를 직접 분석하여 객체를 판단해야 합니다.

또한 다음과 같은 세부 식별 정보는 일반적인 객체 분류보다 더 강한 시각적 증거를 요구하도록 설계했습니다.

- 브랜드
- 제품명
- 정확한 모델명

---

## 사용자 조작

| 입력 | 기능 |
|---|---|
| 마우스 왼쪽 클릭 | 탐지된 객체 선택 및 식별 요청 |
| 마우스 드래그 | 수동 ROI 생성 및 식별 요청 |
| `C` | 현재 선택된 객체 재식별 |
| `S` | 전체 카메라 프레임 저장 |
| `R` | 선택 객체 및 수동 ROI 초기화 |
| `L` | Detector Label 표시 ON/OFF |
| `V` | 선택 객체 Segmentation Mask 표시 ON/OFF |
| `I` | VLM 결과 정보 카드 표시 ON/OFF |
| `Q` | 프로그램 종료 |
| `Esc` | 프로그램 종료 |

기본 설정에서는 객체를 선택하는 즉시 자동으로 VLM 요청이 생성됩니다.

---

## 프로젝트 구조

```text
.
├── run_integrated.py
├── run_integrated.bat
├── integrated_config.json
│
├── yoloe_desktop_pipeline_v2_1.py
├── yoloe_desktop_pipeline_v2_2_ui.py
├── botsort_yoloe_desktop.yaml
│
├── vlm_harness_qwen25vl_v1/
│   ├── harness.py
│   ├── provider.py
│   ├── harness_config.json
│   ├── vision_response.schema.json
│   ├── requirements.txt
│   └── smoke_test.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── EXTENDING.md
│
└── capture/
    └── 실행 중 생성되는 출력 데이터
```

---

## 실행 환경

현재 구현은 주로 Windows 환경에서 개발 및 테스트되었습니다.

주요 소프트웨어 구성요소:

- Python
- PyTorch
- Ultralytics
- OpenCV
- NumPy
- llama.cpp
- Qwen2.5-VL

CUDA를 사용할 수 있는 PyTorch 환경에서는 자동으로 GPU 가속을 사용합니다.

CUDA GPU를 사용할 수 없는 경우 CPU 기반 FP32 추론으로 자동 전환됩니다.

---

## Local VLM 설정

기본 통합 설정에서는 로컬 `llama.cpp` 서버를 사용합니다.

설정 예시:

```json
{
  "local_vlm_server": {
    "enabled": true,
    "exe": "C:\\llama.cpp\\llama-server.exe",
    "repo": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
    "alias": "qwen2.5-vl-3b",
    "host": "127.0.0.1",
    "port": 8080,
    "gpu_layers": 99,
    "context": 4096
  }
}
```

`integrated_config.json`의 `exe` 경로는 사용자의 `llama.cpp` 설치 경로에 맞게 수정해야 합니다.

---

## 통합 파이프라인 실행

Windows:

```bat
run_integrated.bat
```

또는:

```bash
python run_integrated.py
```

통합 실행기는 다음 순서로 동작합니다.

```text
1. 실행 환경 및 필수 파일 확인
2. 기존 llama.cpp 서버 확인 또는 새 서버 실행
3. VLM Harness Worker 실행
4. Camera / YOLOE / HUD 프로세스 실행
5. 프로세스 상태 관리
6. Vision UI 종료 시 하위 프로세스 종료
```

프로젝트 루트에 가상환경이 존재할 경우 다음 Python 실행 파일을 우선 사용합니다.

```text
.venv\Scripts\python.exe
```

---

## 카메라 선택

기본 카메라 인덱스는 다음과 같습니다.

```text
0
```

`integrated_config.json`에서 수정하거나 실행 시 다른 카메라 인덱스를 지정할 수 있습니다.

사용 가능한 카메라 인덱스를 검색하려면:

```bash
python yoloe_desktop_pipeline_v2_2_ui.py --scan
```

을 실행합니다.

---

## 실행 결과 데이터

캡처 및 VLM 요청 관련 데이터는 `capture/` 디렉터리에 저장됩니다.

각 요청에는 다음과 같은 정보가 포함될 수 있습니다.

- 원본 이미지
- VLM 입력 이미지
- Bounding Box 정보
- Segmentation 정보
- 객체 식별자
- Capture 식별자
- Request 식별자
- Detector 메타데이터
- VLM 요청 상태
- 검증된 VLM 응답
- Provider Latency
- 검증 경고
- 오류 정보

주요 요청 상태는 다음과 같습니다.

```text
pending
processing
completed
uncertain
unknown
failed
timeout
cancelled
```

---

## 비동기 처리 안전성

본 프로젝트의 핵심 설계 원칙 중 하나는 다음과 같습니다.

> VLM 결과는 현재 사용자가 보고 있는 객체의 최신 요청에 해당할 때만 UI에 표시해야 합니다.

각 Result Card는 다음 정보를 관리합니다.

```text
object_id
capture_id
request_id
```

이전 요청이 더 늦게 완료되더라도, 이후에 새로운 요청이 생성된 상태라면 오래된 결과는 현재 객체 UI에 연결되지 않습니다.

또한 VLM의 Raw Output은 검증을 통과하기 전까지 사용자에게 직접 표시하지 않습니다.

---

## 현재 한계

현재 버전은 아직 프로토타입 단계입니다.

주요 한계는 다음과 같습니다.

- OpenCV 기반 Desktop UI 사용
- 파일 기반 IPC 사용
- 단일 Local VLM Provider 구현
- Manual ROI는 현재 Tracking되지 않음
- Webcam 중심의 Camera 입력 구조
- 하드웨어 모션 제어 미구현
- 현재 버전에서는 VR/XR Renderer 미통합

---

## 향후 계획

현재 다음 기능들을 개발 또는 검토하고 있습니다.

- Camera Abstraction Layer
- 독립적인 Vision Module
- 독립적인 Motion Control Module
- 교체 가능한 VLM Provider Interface
- Provider Factory
- Desktop Renderer 분리
- HTTP / WebSocket 기반 IPC
- 수동 선택 객체 Tracking 개선
- SAM 기반 Region Tracking
- 하드웨어 Pan/Tilt 연동
- XR / VR HUD 연동
- Spatial Object Anchoring
- VLM 검증 및 Confidence 처리 개선
- 추가 Detection / Segmentation Backend 지원

장기적인 목표는 **Vision, VLM, UI, Transport, Motion Hardware를 서로 독립적으로 유지하는 것**입니다.

이를 통해 특정 카메라, 모델, VLM, UI 또는 구동 하드웨어를 교체하더라도 시스템 전체를 다시 설계하지 않도록 하는 것을 목표로 합니다.

---

## 개발 배경

본 프로젝트의 초기 버전은 **2026년 7월 20일 ~ 2026년 8월 20일** 동안 XR AI 프로젝트의 Vision/Backend 구성요소로 개발되었습니다.

초기 시스템의 주요 목표는 다음과 같았습니다.

- 카메라 영상에서 객체 탐지
- 사용자의 객체 선택
- 이미지 및 메타데이터 저장
- 선택된 이미지를 로컬 Multimodal Model로 전달
- 분석 결과를 XR 환경에 표시

현재는 해당 비전/백엔드 부분을 독립 프로젝트로 발전시키고 있으며 다음 계층을 명확하게 분리하는 방향으로 확장하고 있습니다.

```text
Vision
VLM
UI
Transport
Hardware Motion
```

이 저장소에는 당시 개발한 Vision/Backend 코드와 이후 추가된 확장 기능들이 포함되어 있습니다.

---

## 문서

상세한 내부 구조는 다음 문서에서 확인할 수 있습니다.

### Architecture

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

다음 내용을 설명합니다.

- 프로세스 경계
- Object / Capture / Request 식별 구조
- File IPC
- VLM 처리 구조
- Validation
- 오래된 결과 처리
- Desktop → VR 구조 전환

### Extending the Project

[`docs/EXTENDING.md`](docs/EXTENDING.md)

다음 내용을 설명합니다.

- VLM Backend 교체
- 새로운 Provider 추가
- UI 교체
- Desktop → VR 전환
- Transport Layer 교체

---

## 라이선스

본 프로젝트는 **GNU Affero General Public License v3.0 (AGPL-3.0)**에 따라 배포됩니다.

자세한 내용은 [`LICENSE`](LICENSE)를 참고하세요.

본 프로젝트에서 사용하는 외부 소프트웨어와 모델은 각각 별도의 라이선스가 적용될 수 있습니다.

주요 외부 구성요소:

- Ultralytics YOLO / YOLOE
- PyTorch
- OpenCV
- llama.cpp
- Qwen2.5-VL

사용자는 각 외부 소프트웨어 및 모델에 적용되는 라이선스와 이용 조건을 확인하고 준수해야 합니다.

---

## 면책 사항

본 저장소는 연구 및 개발 목적의 프로젝트입니다.

VLM이 반환하는 객체 식별 결과는 항상 정확한 사실로 간주해서는 안 됩니다.

이미지 품질, 가시성, 촬영 각도, 모델 성능 등에 따라 다음 정보가 부정확할 수 있습니다.

- 객체 종류
- 브랜드
- 제품명
- 모델명

본 프로젝트의 Validation Layer는 근거가 부족한 식별 결과를 줄이기 위한 목적으로 설계되었지만, 결과의 정확성을 보장하지는 않습니다.
