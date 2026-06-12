# AI CUP Table Tennis Sequence Prediction

本專案整理自 AI CUP 桌球賽事序列預測競賽的模型實驗程式碼，主要目標是根據比賽過程中的序列資料，預測多個比賽事件相關任務。

本 repo 並非單一最終模型，而是保留競賽過程中的多階段實驗紀錄，包含從早期 YOLO-like baseline、三頭 GRU 多任務模型、三個任務分開訓練，到後期使用 AutoGluon 進行 tabular ensemble 的完整嘗試。

---

## Project Overview

本專案的核心問題是桌球比賽序列預測。
模型需要根據前綴序列與比賽特徵，預測後續事件或狀態。

競賽任務主要可拆成三個子任務：

* `action`：預測擊球動作或事件類別
* `point`：預測得分相關狀態
* `servegetpoint`：預測發球方或接發球方得分相關結果

本專案的實驗方向依序為：

```text
YOLO-like baseline
→ Three-headed GRU multi-task model
→ Separate task-specific models
→ AutoGluon tabular ensemble
```

---

## Repository Structure

```text
.
├── yolo_like/
│   ├── yolo_like.py
│   ├── yolo_like_fixed.py
│   ├── yolo_like_v2.py
│   ├── yolo_like_v2.2.py
│   └── ...
│
├── cchuang_three-headed/
│   ├── cchuang_gru.py
│   ├── cchuang_v2.py
│   ├── cchuang_v3.1.py
│   ├── cchuang_v4.py
│   ├── cchuang_v5_4priority_noid.py
│   ├── cchuang_v5_crt_ldam_hier_ovr_kmeans_phase_proto.py
│   └── ...
│
├── cchuang_action/
│   ├── cchuang_action_v3.1.py
│   ├── cchuang_action_v3.2.py
│   └── ...
│
├── cchuang_point/
│   ├── cchuang_point_v3.py
│   ├── cchuang_point_v3.1.py
│   ├── cchuang_point_v3.4.py
│   └── ...
│
├── cchuang_serve/
│   ├── cchuang_serve.v3.py
│   ├── cchuang_serve_v3.1.py
│   ├── cchuang_serve_v3.1.1.py
│   └── ...
│
├── autogluon/
│   ├── autogluon_1.py
│   ├── autogluon_2.py
│   ├── autogluon_2_parallel_allmodels_audit.py
│   ├── autogluon_2_taskaware_practical.py
│   └── ...
│
└── README.md
```

---

## Experiment Timeline

### 1. YOLO-like Baseline

第一階段以 YOLO-like 的思路建立 baseline。

此階段主要目的不是建立最強模型，而是快速完成：

* 資料讀取與格式確認
* 序列樣本建立
* 基礎特徵轉換
* 初步模型訓練流程
* submission 輸出格式測試

這個階段幫助確認整體競賽 pipeline 是否可行。

---

### 2. Three-headed GRU Multi-task Model

第二階段改用 GRU 建立三頭多任務模型。

模型設計概念為：

```text
shared sequence encoder
→ action head
→ point head
→ servegetpoint head
```

此階段的核心想法是：

* 使用 GRU 處理比賽序列資訊
* 共用序列表徵
* 針對三個任務分別建立 output head
* 讓模型同時學習多個相關任務
* 測試 multi-task learning 是否能提升整體分數

此階段主要程式位於：

```text
cchuang_three-headed/
```

---

### 3. Task-specific Models

第三階段將三個任務分開訓練，分別針對 `action`、`point`、`servegetpoint` 建立模型。

分開訓練的原因是三個任務的難度、類別分布與特徵需求不同。
若全部任務共用同一個模型，可能會出現部分任務被犧牲或學習不足的情況。

此階段分成：

```text
cchuang_action/
cchuang_point/
cchuang_serve/
```

主要實驗方向包含：

* 任務別特徵工程
* 類別不平衡處理
* label embedding
* sliding prefix features
* allowed history / rule-based bias
* hierarchical prediction logic
* task-specific model tuning

---

### 4. AutoGluon Tabular Ensemble

第四階段使用 AutoGluon 進行 tabular model ensemble。

此階段的目標是利用 AutoGluon 快速比較多種傳統機器學習模型，並透過 ensemble 提升穩定性。

主要實驗內容包含：

* 將序列資料轉換為 tabular features
* 針對不同任務建立 AutoGluon pipeline
* 測試多模型集成
* 嘗試平行化與 practical resume
* 針對任務特性調整訓練方式

此階段主要程式位於：

```text
autogluon/
```

---

## Main Techniques

本專案使用或嘗試過的技術包含：

* Sequence modeling
* GRU
* Multi-task learning
* Three-headed neural network
* Task-specific modeling
* Feature engineering
* Sliding prefix features
* Label embedding
* Class imbalance handling
* Hierarchical prediction
* HMM / state-based features
* Dirichlet smoothing
* KMeans / phase features
* Prototype features
* XGBoost probability features
* AutoGluon ensemble learning

---

## Evaluation

競賽模型主要針對三個任務分別計算表現，並整合成 overall score。

一般流程為：

```text
train data
→ feature engineering
→ model training
→ validation prediction
→ task-specific evaluation
→ test prediction
→ submission generation
```

本專案保留多個版本的原因，是為了記錄不同模型架構與特徵工程對分數的影響。

---

## Requirements

建議使用 Python 3.9 以上版本。

可能使用到的主要套件包含：

```text
pandas
numpy
scikit-learn
torch
xgboost
lightgbm
catboost
autogluon
hmmlearn
matplotlib
tqdm
```

可依實際執行檔案需求安裝：

```bash
pip install pandas numpy scikit-learn torch xgboost lightgbm catboost matplotlib tqdm hmmlearn
```

若要執行 AutoGluon 相關程式，需另外安裝：

```bash
pip install autogluon
```

> AutoGluon 安裝時間較長，且對 Python 版本與環境相容性較敏感，建議建立獨立 conda environment 後再安裝。

---

## How to Run

由於本 repo 保留多階段競賽實驗程式，不同版本的資料路徑與參數設定可能不同。
建議依照想重現的實驗階段進入對應資料夾執行。

例如：

### YOLO-like baseline

```bash
python yolo_like/yolo_like.py
```

### Three-headed GRU

```bash
python cchuang_three-headed/cchuang_gru.py
```

### Task-specific models

```bash
python cchuang_action/cchuang_action_v3.2.py
python cchuang_point/cchuang_point_v3.4.py
python cchuang_serve/cchuang_serve_v3.1.1.py
```

### AutoGluon

```bash
python autogluon/autogluon_2_taskaware_practical.py
```

---

## Notes

本專案是競賽過程中的實驗紀錄，因此保留了多個版本的程式。
不同版本可能對應不同特徵工程、模型架構、參數設定與 submission 嘗試。

若要作為正式可重現專案，建議後續可進一步整理為：

```text
src/
configs/
scripts/
notebooks/
outputs/
```

並將資料路徑、模型參數與任務設定集中到 config file。

---

## Limitations

本專案仍有以下限制：

* 部分程式仍為競賽過程中的實驗版本
* 不同版本的資料路徑可能需要手動調整
* 尚未整理成單一可重現 pipeline
* 部分模型需要較高運算資源
* AutoGluon 相關實驗可能受環境版本影響
* 原始競賽資料不一定包含於此 repo 中

---

## Future Improvements

未來可進一步改善方向包括：

* 整理統一的資料前處理 pipeline
* 將三個任務共用的特徵工程模組化
* 建立 config-based training workflow
* 移除重複或過舊版本程式
* 補充 validation score 與各版本結果比較表
* 建立一鍵產生 submission 的 script
* 將 AutoGluon 與 neural network pipeline 整合
* 加入更完整的 experiment tracking

---

## Disclaimer

This repository is for educational, research, and competition record purposes only.
The code reflects iterative experiments conducted during the AI CUP competition and may require path or environment adjustments before execution.
