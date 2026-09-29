<img width="2560" height="1380" alt="pepar" src="https://github.com/user-attachments/assets/f5dccf31-6faa-4885-8676-e2330cf55db1" />

# PEPAR

## 프로젝트 소개

- arxiv 등지에 배포된 논문과 같은 영문 자료를 번역을 해야할 때가 있었음.
- 웹사이트 부문은 기존 사용하던 papago가 현재 일부 유료 서비스로 전환되었음.
- 크롬 자체 번역은 맥락을 무시하며 번역의 질을 낮추는 체감이 있었음.
- 이에 대한 대안으로 로컬 LLM 사용을 시도해 극복하고자 하는 시도.
- 로컬 LLM의 장점 중 하나인, 비용 문제에서 자유로운 특징을 활용하고자 함.

## 구조

<img width="2000" height="1330" alt="pipeline" src="https://github.com/user-attachments/assets/56da5dc2-2d1c-4ed5-a1ad-62c19ea475ec" />

- 백엔드: Python 3.11, Flask, requests, BeautifulSoup + lxml
- LLM: llama.cpp 라우터(SYCL) + Intel Arc Pro B50 16GB, gemma4-12b 권장
- 저장: DB 없이 파일 캐시 (모델별 번역 HTML, 메타, 체크포인트)
- 운영: systemd 사용자 서비스, Tailscale 내부망 전용

## 과제

- [x] 로컬 LLM 서버(Ollama -> llama server) 정비
- [x] 핵심 기능 구현 계획 수립 및 구현
  - [x] 웹페이지 크롤링
  - [x] 로컬 대형 텍스트 모델이 번역하기 좋은 크기로 슬라이싱
  - [x] arxiv 논문 번역 테스트
  - [x] 로컬 대형 텍스트 모델 선택 버튼
  - [x] 번역한 문서 검색하여 찾기
  - [x] 번역중인 문서 작업 진행도 표시(ETA, 큐 상태)
  - [x] 실패 블록만 재번역, 번역 문서 삭제
  - [x] (개인 운영)systemd 서비스 등록
  - [ ] 번역할 언어 선택
  - [ ] llama.cpp 연동과 포트 설정을 담을 config.json
  - [ ] 보안 수단 검토
- [ ] 배포

## 참고 자료

- Claude 대화 세션
- llama.cpp: https://github.com/ggml-org/llama.cpp
- 프롬프트 캐시(cache-ram) 관련: https://github.com/ggml-org/llama.cpp/pull/16391
- arXiv HTML (LaTeXML 변환): https://info.arxiv.org/about/accessible_HTML.html

## 실행 방법

​```
TBD ASAP
​```

---

## 주차별 기록

### 1주차(2026-09-02 ~ 2026-09-09)

<br>

- 이번 주에 한 일:
  - 챗봇 프로젝트 대비 개인 서버 정비
    - Ollama Gemma4 12b, 26b 세팅 및 api 호출 테스트.
    - 보안상 tailscale 네트워크로만 연결(추후 임시 포트포워딩 & 개방 등 방안 고려)
  - github 레포 관리
- 새롭게 알게 된 것:
  - 로컬 ai의 활용 방안 탐색:
    - 프라이버시, 폐쇄망 환경 적합성 보안, 토큰 비용 등.
    - RAG 등의 단점 보완할 기법 탐색
- 어려웠던 점:
  - 인텔 gpu 특유의 호환성 문제
- 다음 주에 할 일:
  - Gemma4 e2b, e4b 추가 적재 
  - 서버 환경 고려하여 올릴 수 있는 모델 추가 탐색

<br>

### 2주차(2026-09-09 ~ 2026-09-16)

- 이번 주에 한 일:
  - llama server 구축(Ollama에서 전환)
  - 서버 환경 정리(용량 확보, H/W 정비 등)
  - 주제 탐색 및 선정(로컬 LLM 활용 논문 번역도구)
- 새롭게 알게 된 것:
  - 로컬 LLM의 장단점(비용, 프라이버시 / 구축 비용)
- 어려웠던 점:
  - 주제 발산
- 다음 주에 할 일:
  - 실제 구현 착수

<br>

### 3주차(2026-09-16 ~ 2026-09-23)

- 이번 주에 한 일:
  - 슬라이싱하여 html 문서 번역 품질 테스트. 백엔드 위주 개발
  - 본 목적대로 arxiv 논문에서 번역 품질 확보
- 새롭게 알게 된 것:
  - LLM에서 KV캐시의 관리
  - Dense와 MoE의 개념
  - H/W 한계 내에서 가동가능한 LLM(ARC B50 16GB -> Dense 14B 이하 & MoE 35B 이하)
- 어려웠던 점:
  - 서버 RAM 한계로 인해 각종 서비스 최적화
  - 번역 중 서버 메모리 부족(OOM)으로 서비스가 강제 종료됨
    - llama-server의 프롬프트 캐시(cache-ram, 기본 8GB)가 원인, 1GB로 제한해 해결
- 다음 주에 할 일:
  - UX/UI 다듬기

<br>

### 4주차(2026-09-23 ~ 2026-09-30)

- 이번 주에 한 일:
  - 프런트엔드 구현 UX/UI 다듬기
  - tailscale 네트워크 상으로만 개방하여 테스트
- 새롭게 알게 된 것:
  - HTML 파싱과 번역 단위 설계: 수식·링크를 태그로 보호하고 번역 후 복원
  - 프롬프트 형식의 영향: 번역과정에서 처리하는 임의 기호(`⟦0⟧`)는 모델이 괄호로 착각 → HTML 태그 형식(`<t0>`)으로 바꿔 해결
  - 제약 디코딩: JSON 스키마를 강제해 출력 형식 오류 제거
  - thinking 모드 끄기만으로 요청 시간 150초 → 13초
- 어려웠던 점:
  - 웹페이지를 가져와 다시 띄우는 방식의 한계: JS 렌더링, CORS, 봇 차단(Cloudflare), 로그인 페이지
- 다음 주에 할 일:
  - html 페이지 본문 번역이 아닌 단순 번역기 역할로도 확장 시도
  - 한국어로 국한하지 않고 번역
  - repo에서 (비공개)배포

<br>
