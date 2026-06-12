# 前提

這是一個玩Codex附加學習實作(?)LLM如何訓練的小專案，內容多由Codex產生外加一些個人的
程式語法習慣修改。

# 從零訓練 Transformer 語言模型

這是一個教學用的 GPT 類 decoder-only Transformer。它不是微調現成模型，而是用
PyTorch 從隨機權重開始訓練，包含 causal self-attention、位置嵌入、殘差連接、
LayerNorm、AdamW、validation、test perplexity、checkpoint 與文字生成。

## 資料集

使用 [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
中的 `wikitext-2-raw-v1`：

- 訓練集：36,718 列
- 驗證集：3,760 列，用於選最佳 checkpoint
- 測試集：4,358 列，只在訓練結束後計算 loss/perplexity
- 授權：CC BY-SA

WikiText-2 體積小、切分固定，適合先驗證完整 LLM 訓練流程。它是英文 Wikipedia
語料，因此這個模型主要學英文；若目標是中文實用模型，應改用更大且授權清楚的中文
語料，並使用 BPE/SentencePiece tokenizer。

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

程式會保留 WikiText 官方的 `train`、`validation`、`test` 切分，並使用 UTF-8
byte-level tokenizer。詞彙表固定為 257 個 token，不需要另外訓練 tokenizer，
且任何語言都不會出現 unknown token。

## 訓練

```bash
python3 train.py
```

程式會自動選擇 CUDA、Apple MPS 或 CPU，並把最佳模型存到
`checkpoints/best.pt`。先做快速測試可執行：

```bash
python3 train.py \
  --max_steps 20 \
  --eval_interval 10 \
  --eval_batches 2 \
  --batch_size 4 \
  --context_length 64 \
  --d_model 64 \
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
python3 generate.py --prompt "The history of artificial intelligence"
```

## 重要限制

預設模型只有數百萬參數，WikiText-2 也只有小型語料，因此這是完整的 LLM
訓練範例，不會得到 ChatGPT 等級的能力。實務級模型還需要更大的去重語料、
更好的 tokenizer、分散式訓練、混合精度、學習率排程、資料品質控制與安全評估。
