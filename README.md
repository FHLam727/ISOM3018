# 資料預處理說明 (Data Preprocessing)

## 概覽

本專案針對 BNPL（先買後付）信用風險資料集進行資料預處理，輸出供所有組員共用的訓練集與測試集。

---

## 原始資料集

- **檔案**：`Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset.csv`
- **大小**：10,345 筆記錄，17 個變數
- **目標變數**：`default_flag`（0 = 未違約，1 = 違約）

---

## 預處理步驟

### 第一步：移除不相關欄位

以下三個欄位於建模前被移除：

| 欄位 | 移除原因 |
|------|----------|
| `user_id` | 僅為識別碼，不具預測能力 |
| `transaction_date` | 本專案不進行時間序列分析 |
| `customer_segment` | **資料洩漏（Data Leakage）**：此欄位由違約行為衍生而來，若保留將導致模型「預先得知答案」，評估結果失真 |

移除後剩餘 **14 個變數**。

### 第二步：轉換資料類型

- `default_flag`：轉換為 `factor`（`No_Default` / `Default`）
- `employment_type`、`product_category`、`location`：轉換為 `factor`
- `bnpl_installments`：轉換為 `numeric`

### 第三步：訓練集與測試集分割（80/20）

- 使用 `set.seed(42)` 確保所有組員的分割結果一致
- 分割比例：訓練集 80%、測試集 20%
- 使用 `strata = default_flag` 進行分層抽樣，確保兩組的目標變數比例相近

| 資料集 | 筆數 |
|--------|------|
| 訓練集（`train_raw`） | 8,276 筆 |
| 測試集（`test_df`） | 2,069 筆 |

### 第四步：類別不平衡處理（SMOTE）

原始資料存在輕度類別不平衡（未違約 61%，違約 39%），採用 SMOTE（Synthetic Minority Over-sampling Technique）處理。

**重要原則：SMOTE 僅施加於訓練集，測試集絕對不進行過採樣。** 若在分割前執行 SMOTE，人工生成的樣本將洩漏至測試集，導致模型評估結果虛假偏高。

- 使用套件：`themis`
- 參數：`over_ratio = 1`（將少數類別過採樣至與多數類別相同數量）、`neighbors = 5`

SMOTE 後訓練集兩類別數量相等。

### 第五步：類別變數編碼（One-Hot Encoding）

使用 `recipes` 套件的 `step_dummy()` 對所有類別型預測變數進行 One-Hot Encoding，訓練集與測試集使用相同的 recipe，確保欄位結構一致。

編碼後共有 **26 個變數**。

---

## 輸出檔案

| 檔案 | 內容 | 大小 |
|------|------|------|
| `train_df.rds` | 經 SMOTE 及 One-Hot Encoding 處理的訓練集 | 10,088 筆，26 個變數 |
| `test_df.rds` | 僅經 One-Hot Encoding 處理的測試集 | 2,069 筆，26 個變數 |

---

## 組員使用方法

所有組員於各自的模型檔案開頭載入以下兩行即可，無需重複執行預處理：

```r
train_df <- readRDS("train_df.rds")
test_df  <- readRDS("test_df.rds")
```

---

## 使用套件

| 套件 | 用途 |
|------|------|
| `tidyverse` | 資料處理 |
| `rsample` | 訓練集／測試集分割 |
| `recipes` | 預處理 Pipeline（One-Hot Encoding） |
| `themis` | SMOTE 過採樣 |

---

## 注意事項

- 各組員於模型層面如需進行特徵縮放（Scaling），請自行在載入資料後處理。KNN 及 Logistic Regression 建議進行標準化，Random Forest、Decision Tree 及 XGBoost 則不需要。
- 請勿對 `test_df` 進行任何額外的過採樣或資料增強操作。
