## Train *Char* Tokenizer

```bash
python -m src.train.train_tokenizer \
    --model_name parakeet-tdt-0.6b-v2 \
    --tokenizer_type char \
    --output_dir data/checkpoints/tokenizer/char
```

---

## Train *SentencePiece* Tokenizer (Unigram)

SentencePiece does not support mini-batch SGD.
For large corpora, use `--input_sentence_size` to randomly sample sentences during training.

```bash
python -m src.train.train_tokenizer \
    --model_name parakeet-tdt-0.6b-v2 \
    --tokenizer_type sentencepiece_unigram \
    --output_dir data/checkpoints/tokenizer/sp_unigram \
    --vocab_size 8192 \
    --input_sentence_size 5000000
```

If `--input_sentence_size` is omitted, the default is `5000000`.
Set `--input_sentence_size 0` to use the full corpus.

---

## Train *SentencePiece* Tokenizer (BPE)

```bash
python -m src.train.train_tokenizer \
    --model_name parakeet-tdt-0.6b-v2 \
    --tokenizer_type sentencepiece_bpe \
    --output_dir data/checkpoints/tokenizer/sp_bpe \
    --vocab_size 8192 \
    --input_sentence_size 5000000
```

---

## Reuse an Existing Corpus

```bash
python -m src.train.train_tokenizer \
    --model_name parakeet-tdt-0.6b-v2 \
    --tokenizer_type sentencepiece_unigram \
    --output_dir data/checkpoints/tokenizer/sp_unigram \
    --corpus_path data/checkpoints/char/corpus_train.txt \
    --vocab_size 8192 \
    --input_sentence_size 5000000
```
