# LG_Aimer_2025
# 📈 Resort Menu Demand Forecasting

리조트 내 식음업장 **메뉴별 매출 수요 예측 시스템**입니다.
Tree 모델, 딥러닝 모델, 통계 모델을 **혼합 앙상블**하여 **+1일 \~ +7일 Horizon**을 예측합니다.

---

## ⚙️ 주요 기능

* **데이터 전처리 & 피처 엔지니어링**

  * Lag, Rolling Mean, Zero Count/Streak 등 시계열 피처 생성
  * 아이템(메뉴)별 카테고리 인코딩
* **모델 학습**

  * 🌳 Tree: **LightGBM**, **XGBoost** (GPU 가속)
  * 🤖 딥러닝: **N-BEATS (아이템 임베딩 포함)** – PyTorch 구현
  * 📉 통계: **SARIMAX (AIC 기반 선택)**, **ZIP (Zero-Inflated Poisson)**
* **앙상블(Ensemble)**

  * XGBoost + LightGBM + N-BEATS + SARIMAX(+ZIP) 결과를 **가중합**
  * `scipy.optimize.minimize` 기반 **sMAPE 최소화 자동 가중치 튜닝**

---

## 🧰 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-004C6D?style=for-the-badge\&logo=lightgbm\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge\&logo=xgboost\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge\&logo=scipy\&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-003366?style=for-the-badge\&logo=python\&logoColor=white)

---

## 📊 주요 하이라이트

* **N-BEATS 모델 직접 구현** (Item Embedding, SMAPE Loss, AMP 지원)
* **Auto Weight Tuning** (LightGBM/XGBoost/N-BEATS/ZIP 가중치 자동 조정)
* **GPU 활용** (PyTorch, LGBM, XGB 학습 가속)

---

## 🏆 결과물 & 성과

* LG Aimers 해커톤 **수료**
* 시계열 + 딥러닝 + 통계 모델을 결합한 앙상블 실험 수행
* <img src="수료증_이미지_경로" width="400px"/>

---

## 🙌 기여 & 연락

* Issue/PR 환영합니다.
* 문의: [your.email@example.com](mailto:your.email@example.com)
