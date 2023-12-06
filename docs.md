# AI CUP 2023 Fall Go

## 執行環境

### 使用 Anaconda 建立環境

```bash
conda create -n go-aicup python=3.10
pip install -r requirements.txt
```

## 資料

- `csv/dan_train.csv`: 官方提供的 dan 訓練資料
- `csv/dan_test_public.csv`: 官方提供的 dan public 測試資料
- `csv/dan_test_private.csv`: 官方提供的 dan private 測試資料
- `csv/kyu_train.csv`: 官方提供的 kyu 訓練資料
- `csv/kyu_test_public.csv`: 官方提供的 kyu public 測試資料
- `csv/kyu_test_private.csv`: 官方提供的 kyu private 測試資料
- `csv/play_style_train.csv_train.csv`: 官方提供的棋風辨識訓練資料
- `csv/play_style_test_public.csv`: 官方提供的棋風辨識 public 測試資料
- `csv/play_style_test_private.csv`: 官方提供的棋風辨識 private 測試資料

資料前處理的程式放在 `src/data` 資料夾中，處理後的檔案會放在 `data` 資料夾中。

src/data/dan_train_10M.py: 分割原始資料集
scripts/generate_span_dataset.py: 由原始資料集生成 Span 資料集
scripts/generate_mutli_target_dataset.py: 由原始資料集生成 Multi-Target 資料集