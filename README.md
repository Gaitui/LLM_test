# 前提

這是一個玩Codex附加學習實作(?)LLM如何訓練的小專案，內容多由Codex產生外加一些個人的
程式語法習慣修改。

# 從零訓練 Transformer 語言模型

這是一個教學用的 GPT 類 decoder-only Transformer。它不是微調現成模型，而是用
PyTorch 從隨機權重開始訓練，包含 causal self-attention、位置嵌入、殘差連接、
LayerNorm、AdamW、validation、test perplexity、checkpoint 與文字生成。

## 資料集

使用 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。
這份資料約有 200 萬篇由 GPT-3.5/GPT-4 產生的簡單英文故事，原研究的目的就是讓
小於 1,000 萬參數的模型也能學會連貫、文法正確的英文，比 WikiText-2 更適合這個
小型教學模型。

- 預設訓練集：1 億 GPT-2 BPE tokens，可用參數繼續放大
- Validation：TinyStories 官方 validation 的偶數篇
- Test：TinyStories 官方 validation 的奇數篇
- Validation 用來選 checkpoint；test 只在訓練完成後評估

資料採串流下載並存成 `uint16` memory-map，1 億 tokens 約占 200 MB。訓練時不會
把整份資料載入 RAM。TinyStories 是合成的簡單英文故事，所以生成品質會提升，但模型
會偏向故事文體，並不等於通用聊天模型。

## 安裝

建議建立虛擬環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 下載與編碼資料

```bash
python3 prepare_data.py
```

程式使用 GPT-2 BPE tokenizer，預設下載並編碼 1 億個訓練 tokens：

```bash
python3 prepare_data.py --max_train_tokens 100000000
```

要準備完整 TinyStories 訓練集，使用 `0` 取消上限：

```bash
python3 prepare_data.py --max_train_tokens 0
```

也可以先用 1,000 萬 tokens 快速確認流程：

```bash
python3 prepare_data.py \
  --max_train_tokens 10000000 \
  --max_eval_tokens 200000
```

## 訓練

```bash
python3 train.py
```

程式會自動選擇 CUDA、Apple MPS 或 CPU，並把最佳模型存到
`checkpoints/tinystories/best.pt`。預設設定約 3,000 萬參數，明顯比舊版更有能力，
但也需要較長訓練時間。先做快速測試可執行：

```bash
python3 train.py \
  --max_steps 20 \
  --eval_interval 10 \
  --eval_batches 2 \
  --batch_size 4 \
  --context_length 128 \
  --d_model 128 \
  --n_layers 2 \
  --n_heads 4
```

若記憶體不足，先降低 `--batch_size`、`--context_length` 或 `--d_model`。
要擴大模型，可提高 `--d_model`、`--n_layers`、`--n_heads`，但訓練時間與所需
資料量也會大幅上升。

## 測試與生成

不下載資料的程式自我測試：

```bash
python3 test_smoke.py
```

訓練後生成文字：

```bash
python3 generate.py --prompt "Once upon a time, Lily found a little blue box"
```

## 重要限制

TinyStories 主要改善小模型的英文故事生成，不會得到 ChatGPT 等級的通用問答能力。
實務級模型還需要更大且多樣的去重語料、
更好的 tokenizer、分散式訓練、混合精度、學習率排程、資料品質控制與安全評估。
