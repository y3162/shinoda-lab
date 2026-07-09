## Train *Char* Tokenizer

```bash
python -m src.train.train_tokenizer \
    --model_name parakeet-tdt-0.6b-v2 \
    --tokenizer_type char \
    --output_dir data/checkpoints/tokenizer/char
```

---

## Train *SentencePiece* Tokenizer (Unigram)

```bash
python -m src.train.train_tokenizer \
    --model_name parakeet-tdt-0.6b-v2 \
    --tokenizer_type sentencepiece_unigram \
    --output_dir data/checkpoints/tokenizer/sp_unigram \
    --vocab_size 8192
```

---

## Train *SentencePiece* Tokenizer (BPE)

```bash
python -m src.train.train_tokenizer \
    --model_name parakeet-tdt-0.6b-v2 \
    --tokenizer_type sentencepiece_bpe \
    --output_dir data/checkpoints/tokenizer/sp_bpe \
    --vocab_size 8192
```
