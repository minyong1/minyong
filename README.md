# 📈 Stock Quant App

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Meta--Labeling-2ECC71)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Web-0078D6)

미국 주식·ETF 및 국내 주식(KOSPI/KOSDAQ)을 대상으로 **Triple-Barrier 라벨링**, **LightGBM Meta-Labeling**, **Bet Sizing(투자 비중 조절)**을 결합해 오늘의 매수 추천 종목과 Walk-Forward 백테스트 결과를 제공하는 퀀트 대시보드입니다.  
또한 증권사 차트를 보면서 참고할 수 있는 **차트 시나리오 도우미(과거 유사 구간 찾기 · 예상 레인지 계산기)**를 지원합니다.

> 💻 **Windows 실행 파일(`.exe`) 다운로드:** [Releases 페이지에서 최신 버전 받기](https://github.com/minyong1/minyong/releases)

---

## ✨ 주요 기능

1. **오늘의 추천 종목 (스캐너)**
   - 관심 종목(기본 20개 또는 시장별 필터)을 일괄 스캔해 1차 상승 추세(`EMA20 > EMA60`)와 2차 LightGBM 성공 확률을 결합합니다.
   - **추천 등급**, **성공 확률(%)**, **추천 투자 비중(%)**, **목표 익절가**, **손절 기준가**를 우선순위대로 자동 정렬합니다.
2. **개별 종목 신호**
   - 사이드바에서 선택한 단일 종목의 오늘 기준 판정(`BUY` 또는 `HOLD`)과 익절·손절 기준 라인을 즉시 조회합니다.
3. **Walk-Forward 백테스트**
   - 시간순 전진 분석(Walk-Forward)과 청산일 기준 경계 정화(Purging)를 적용해 미래 데이터 누수를 차단합니다.
   - `Buy & Hold`, `1차 추세 단독`, `Meta-Labeling + Bet Sizing` 3가지 전략의 누적 수익률, CAGR, Sharpe Ratio, MDD, 승률을 비교합니다.
4. **차트 시나리오 도우미 (`v1.1` 추가)**
   - **① 과거 유사 구간 찾기:** 최근 $N$일 가격 흐름과 유사한 과거 구간(상위 $K$개)을 찾아 이후 $M$일간의 평균 수익률, 상승 비율, 시나리오 가격 범위를 시각화합니다.
   - **② 예상 레인지 계산기:** 현재가와 일별 변동성(%), 예상 기간을 입력해 기간별 예상 상단·하단 가격 범위를 계산합니다.
5. **한·미 거시 지표 시차 보정 (Look-ahead Bias 차단)**
   - `S&P 500`, `VIX`, `미국 10년물 국채금리`, `KOSPI`, `원/달러 환율`을 자동 결합하며, 국내 종목 분석 시 미국 시장 마감 시차를 달력 기준 `+1일` 보정합니다.
6. **주간 자동 재학습 & 단독 실행(`.exe`) 지원**
   - Windows 작업 스케줄러 연동용 CLI(`weekly_job.py`)와 `signals_log.csv` 자동 누적 기록을 지원합니다.

---

## 📖 화면별 사용법

### 1단계: 사이드바에서 종목 및 전략 설정
- **빠른 종목 선택:** 기본 등록된 미국·한국 대표 20개 종목 중 선택하거나, **`직접 입력`**을 골라 Yahoo Finance 티커(예: `PLTR`, `005380.KS`)를 입력합니다.
- **장벽 및 진입 기준 조절:**
  - `익절 / 손절 장벽 배수`: 최근 20일 일별 변동성에 배수를 곱해 목표 익절가와 손절 기준가를 산출합니다.
  - `2차 진입 기준 확률` (기본 `0.52`): Meta-Model 성공 확률이 이 기준을 넘을 때만 매수 비중이 배분됩니다.

### 2단계: 탭별 기능 활용하기
- **전체 시장 훑어보기 (`오늘의 추천 종목 (스캐너)`):**
  1. 스캔할 시장(`전체 (20개)`, `미국 개별주`, `한국 코스피` 등)을 고릅니다.
  2. **`스캔 및 추천 보기`** 버튼을 누르면 `추천비중(%)`과 `성공확률(%)`이 높은 순으로 표가 정렬됩니다.
- **단일 종목 정밀 확인 (`개별 종목 신호`):**
  - 평소에는 **`오늘 매수 신호 확인 (저장 모델 사용)`**으로 빠르게 조회하고, 최신 시장 데이터를 반영하고 싶을 때 **`최신 데이터로 지금 재학습 후 확인`**을 누릅니다.
  - 판정이 `HOLD (2차 필터 미달)`이면 상승 추세는 유지 중이나 성공 확률이 기준치에 못 미쳐 관망하는 구간을 의미합니다.
- **전략 검증 (`Walk-Forward 백테스트`):**
  - 재학습 주기(21거래일 또는 5거래일)를 고른 뒤 **`Walk-Forward 백테스트 실행`**을 누르면 `Buy & Hold` 대비 전략 성과표, 누적 수익률 곡선, 피처 중요도, 최근 매매 내역을 확인할 수 있습니다.
- **증권사 차트 보조 분석 (`차트 시나리오 도우미`):**
  - **`① 과거 유사 구간 찾기` 버튼:** 비교할 최근 기간(거래일)과 이후 확인할 기간(거래일)을 설정하고 **`유사 구간 검색 실행`**을 누릅니다. 조회된 현재가와 일별 변동성은 다음 계산기 모드에 자동 연동됩니다.
  - **`② 예상 레인지 계산기` 버튼:** 증권사 차트에서 확인한 현재가, 일별 변동성(%), 예상 기간(거래일)을 입력하고 **`예상 레인지 계산`**을 누르면 예상 상단·하단 가격 범위를 확인할 수 있습니다.

---

## 📊 기본 지원 종목 (총 20개)

| 구분 | 기본 포함 종목 (티커) |
| :--- | :--- |
| **미국 개별주 (7)** | 애플(`AAPL`), 엔비디아(`NVDA`), 마이크로소프트(`MSFT`), 테슬라(`TSLA`), 아마존(`AMZN`), 알파벳(`GOOGL`), 메타(`META`) |
| **미국 ETF (4)** | S&P500(`SPY`), 나스닥100(`QQQ`), 반도체(`SOXX`), 미국 장기국채(`TLT`) |
| **한국 코스피 (6)** | 삼성전자(`005930.KS`), SK하이닉스(`000660.KS`), NAVER(`035420.KS`), 현대차(`005380.KS`), LG에너지솔루션(`373220.KS`), KODEX 200(`069500.KS`) |
| **한국 코스닥 (3)** | 에코프로비엠(`247540.KQ`), 알테오젠(`196170.KQ`), 에코프로(`086520.KQ`) |

> 사이드바에서 **`직접 입력`**을 선택하면 Yahoo Finance(`yfinance`)가 지원하는 전 세계 주식 및 ETF를 자유롭게 추가 분석할 수 있습니다.

---

## 🚀 설치 및 실행 방법

### 방법 1. Windows `.exe` 파일로 바로 실행 (Python 설치 불필요)
1. [Releases](https://github.com/minyong1/minyong/releases)에서 최신 버전의 `StockQuantApp.zip`을 다운로드해 압축을 풉니다.
2. 압축을 푼 폴더 안의 `StockQuantApp.exe`를 더블클릭하면 웹 브라우저에 대시보드가 자동으로 열립니다.
   - `_internal` 폴더가 반드시 `StockQuantApp.exe`와 같은 폴더 안에 있어야 정상 작동합니다.
   - 처음 실행 시 **Windows의 PC 보호** 경고 창이 뜨면 **`추가 정보` → `실행`**을 클릭하세요.

### 방법 2. 소스 코드로 직접 실행 (개발자용)
```bash
git clone https://github.com/minyong1/minyong.git
cd minyong

# 가상환경 생성 및 패키지 설치 (Conda 권장)
conda create -n stock_quant python=3.12 -y
conda activate stock_quant
pip install -r requirements.txt

# 웹 대시보드 실행
streamlit run dashboard_app.py
```

### 터미널 일괄 재학습 및 신호 기록 (CLI)
```bash
# 최신 데이터로 강제 재학습 후 전체 종목 신호 기록 (signals_log.csv 생성)
python weekly_job.py --retrain

# 저장된 모델로 빠르게 오늘 신호만 확인
python weekly_job.py
```

---

## ⚙️ 대시보드 파라미터 가이드

| 파라미터 | 기본값 | 설명 |
| :--- | :---: | :--- |
| **익절 장벽 배수** | `2.00` | 최근 20일 일별 변동성 × 배수로 목표 익절가 자동 산출 |
| **손절 장벽 배수** | `1.50` | 최근 20일 일별 변동성 × 배수로 손절 기준가 자동 산출 |
| **최대 보유 거래일** | `10` | 익절·손절 미도달 시 강제 청산하는 최대 보유 기간 |
| **2차 진입 기준 확률** | `0.52` | Meta-Model 성공 확률이 이 값을 넘을 때만 매수 진입 승인 |
| **비중 조절 방식** | `afml` | `afml`(표준정규 CDF 기반 보수적 배분) / `linear`(선형 배분) |

---

## 🛠️ 패키징 및 자동화

### PyInstaller 단독 실행 파일(`.exe`) 빌드
```bash
pip install pyinstaller

pyinstaller --onedir --noconfirm --name StockQuantApp --add-data "dashboard_app.py;." --collect-all streamlit --copy-metadata streamlit --collect-all lightgbm --collect-all yfinance --collect-all curl_cffi --collect-all certifi --hidden-import scipy.special.cython_special run_launcher.py
```

### Windows 작업 스케줄러 등록 예시 (PowerShell)
```powershell
# 매주 토요일 오전 10:00 강제 재학습
schtasks /create /tn "StockQuant_WeeklyRetrain" /tr "C:\경로\StockQuantApp\StockQuantApp.exe --weekly --retrain" /sc weekly /d SAT /st 10:00 /f

# 평일 오전 08:30 오늘의 매수 신호 기록
schtasks /create /tn "StockQuant_DailySignal" /tr "C:\경로\StockQuantApp\StockQuantApp.exe --weekly" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:30 /f
```

---

## 📁 프로젝트 구조

```text
stock_quant_app/
├── core_engine.py       # 데이터 수집, 피처 생성, Triple-Barrier, Meta-Labeling, 시나리오 분석 엔진
├── dashboard_app.py     # Streamlit 웹 대시보드 UI (스캐너, 개별 신호, 백테스트, 로그, 시나리오 도우미)
├── weekly_job.py        # 20개 종목 일괄 재학습 및 signals_log.csv 기록 스크립트
├── run_launcher.py      # PyInstaller .exe 진입점 (대시보드 실행 및 --weekly CLI 분기)
└── requirements.txt     # 필수 Python 패키지 목록
```

---

## ⚠️ 면책 조항 (Disclaimer)

본 프로젝트는 머신러닝 및 퀀트 전략 연구·학습 목적으로 제작되었습니다. 제시되는 매수 추천, 성공 확률, 목표가, 시나리오 분석 및 백테스트 성과는 과거 주가와 거시 지표에 기반한 통계적 산출물이며 실적 발표·공시·뉴스 등의 변수는 반영되지 않습니다. 모든 투자 판단과 그에 따른 손익의 책임은 사용자 본인에게 있습니다.