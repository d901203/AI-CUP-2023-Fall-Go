# AI CUP 2023 Fall Go

## 執行環境

使用 Anaconda 建立環境

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

**處理程式**

資料前處理的程式放在 `src/data` 資料夾中，會讀取 `csv` 資料夾中的檔案，處理完的檔案會放在 `data` 資料夾中。

- `src/data/utils.py`: 一些共用的函式
- `src/data/dan_train_10M.py`: 生成 dan 10M 訓練資料，儲存在 `data/dan/10M` 資料夾中
- `src/data/dan_train_20M.py`: 生成 dan 20M 訓練資料，儲存在 `data/dan/20M` 資料夾中
- `src/data/dan_test_public.py`: 生成 dan public 測試資料，儲存在 `data/dan/test` 資料夾中
- `src/data/dan_test_private.py`: 生成 dan private 測試資料，儲存在 `data/dan/test` 資料夾中
- `src/data/kyu_train_10M.py`: 生成 kyu 10M 訓練資料，儲存在 `data/kyu/10M` 資料夾中
- `src/data/kyu_train_20M.py`: 生成 kyu 20M 訓練資料，儲存在 `data/kyu/20M` 資料夾中
- `src/data/kyu_test_public.py`: 生成 kyu public 測試資料，儲存在 `data/kyu/test` 資料夾中
- `src/data/kyu_test_private.py`: 生成 kyu private 測試資料，儲存在 `data/kyu/test` 資料夾中
- `src/data/play_style_train.py`: 生成棋風辨識訓練資料，儲存在 `data/play_style` 資料夾中
- `src/data/play_style_test_public.py`: 生成棋風辨識 public 測試資料，儲存在 `data/play_style/test` 資料夾中
- `src/data/play_style_test_private.py`: 生成棋風辨識 private 測試資料，儲存在 `data/play_style/test` 資料夾中

**註:** 
- 生成訓練資料時，可以修改程式中 NUM_CORES 的數值。
- 生成訓練資料時，可以修改程式中 BATCH_SIZE 的數值，
dan 和 kyu 會根據 BATCH_SIZE 的數值分成多個檔案，棋風辨識則不會。

## 模型架構

- `src/model/dan_model.ipynb`: dan 模型架構
- `src/model/kyu_model.ipynb`: kyu 模型架構
- `src/model/play_style_model.ipynb`: 棋風辨識模型架構

## 訓練

會讀取 `data` 資料夾中的檔案，訓練完的模型會放在 `checkpoints` 資料夾中。

- `train_dan_10M.py`: 訓練 dan 10M 模型
- `train_dan_20M.py`: 訓練 dan 20M 模型
- `train_kyu_10M.py`: 訓練 kyu 10M 模型
- `train_kyu_20M.py`: 訓練 kyu 20M 模型
- `train_play_style.py`: 訓練棋風辨識模型

**註:**
- 程式內的模型為測試後的最佳模型，可以修改程式中的模型架構。
- 訓練 dan 或 kyu 時，要確認 `data/dan/10M (20M)` 和 `data/kyu/10M (20M)` 資料夾中的訓練資料數量。可以在 `load_data` 函式中修改。
- 訓練 dan 或 kyu 的 20M 模型時，根據我們的訓練方式，會先載入訓練好的 10M 模型。
- 訓練 dan 或 kyu 時，若在訓練後段出現 `nan` 的錯誤，不會影響結果，即完成訓練。
- 不建議開啟 `Automatic Mixed Precision (AMP)` 功能，會導致訓練開始就出現 `nan` 的錯誤。 

## 預測

會讀取 `data` 資料夾中的檔案，預測完的結果會放在根目錄中。

- `predict_dan_kyu.py`: 預測 dan 和 kyu
- `predict_play_style.py`: 預測棋風辨識

**註:**
- 程式內的模型為測試後的最佳模型，可以修改程式中的模型架構。
- 確認 `data/*/test` 資料夾中是否有測試資料，可以透過 `src/data` 資料夾中的程式生成。
- 預測 dan 或 kyu 時，可以修改 `A` 參數，決定要預測的是 `public` 或 `private` 測試資料。
- 預測 dan 或 kyu 時，可以修改 `B` 參數，決定要預測的是 `dan` 或 `kyu`。
- 預測 dan 或 kyu 時，可以修改 `MODEL` 參數，決定載入的模型。
- 預測棋風辨識時，可以修改 `A` 參數，決定要預測的是 `public` 或 `private` 測試資料。
- 預測棋風辨識時，可以修改 `MODEL` 參數，決定載入的模型。

## 模型權重

- `weights/dan`: dan 模型權重
- `weights/kyu`: kyu 模型權重
- `weights/play_style`: 棋風辨識模型權重

**註:**
- 模型架構可以參考 `src/model` 資料夾中的程式。
- 載入棋風辨識模型時，要先將模型放置在 `DataParallel` 中，再載入權重。

**kyu**

| id | Pre-Activation Policy Head | Loss Function | Depth | Main Channels | 訓練集 | Public Top1 | Public Top5 | Public 加權 | Private |
| :---: | :------------------------: | :-----------: | :---: | :-----------: | :----: | :---------: | :---------: | :---------: | :-----: |
| A |             N              | Cross-Entropy |   5   |      192      |  10M   |  0.565873    |  0.861640   |  0.22763225  |    X    |
| B |             N              |  Focal Loss   |   5   |      192      |  20M   |  0.568519   |  0.863404   | 0.22847015  |    X    |
| C |             N              | Cross-Entropy |   5   |      192      |  20M   |  0.574074   |  **0.865168**   |  0.23003530  |    **1**    |
| D |             N              |  Cross-Entropy   |   7   |      192      |  10M   |  0.568342   |  0.863051   | 0.22839060  |    X    |
| E |             N              |  Cross-Entropy  |   7   |      192      |  20M   |   0.571958    |  0.863757   | 0.22936520  |    2    |
| F |             Y              |  Cross-Entropy  |   5   |      192      |  10M   |  0.571693   |  0.861552  | 0.22907845  |    4    |
| G |             Y              |  Cross-Entropy   |   5   |      192      |  20M   |  **0.575661**   |  0.862257   | **0.23014095**  |    3    |

**dan**

| id | Pre-Activation Policy Head | Loss Function | Depth | Main Channels | 訓練集 | Public Top1 | Public Top5 | Public 加權 | Private |
| :---: | :------------------------: | :-----------: | :---: | :-----------: | :----: | :---------: | :---------: | :---------: | :-----: |
| A |             N              | Cross-Entropy |   5   |      192      |  10M   |    0.554    |  0.862273   | 0.22472730  |    X    |
| B |             N              |  Focal Loss   |   5   |      192      |  20M   |  0.563545   |  0.867909   | 0.22767715  |    3   
| C |             N              | Cross-Entropy |   5   |      192      |  20M   |  0.559091   |  0.865727   | 0.22634545  |    X    |
| D |             N              |  Focal Loss   |   7   |      192      |  10M   |  0.558273   |  0.867182   | 0.22628645  |    X    |
| E |             N              |  Focal Loss   |   7   |      192      |  20M   |    0.562    |  **0.869818**   | 0.22748180  |    2    |
| F |             Y              |  Focal Loss   |   5   |      192      |  10M   |  0.557909   |  0.865182   | 0.22599545  |    4    |
| G |             Y              |  Focal Loss   |   5   |      192      |  20M   |  **0.567909**   |  0.869545   | **0.22893175**  |    **1**    |

**棋風辨識**

| id | Pre-Activation Policy Head | Depth | Main Channels |  Public  | Private |
| :---: | :------------------------: | :---: | :-----------: | :------: | :-----: |
| A |             N              |   5   |      192      | 0.809224 |    4    |
| B |             N              |   7   |      192      | 0.793617 |    X    |
| C |             Y              |   5   |      192      | 0.812225 |    3    |
| D |             Y              |   5   |      192      | 0.815926 |    2    |
| E |             Y              |   5   |      256      | **0.82593** |    **1**     |
