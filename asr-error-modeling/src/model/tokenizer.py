from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
SPECIAL_IDS = (PAD_ID, BOS_ID, EOS_ID, UNK_ID)

CONFIG_FILE = 'config.json'
CHAR_TOKENIZER_FILE = 'tokenizer.json'
SP_MODEL_FILE = 'tokenizer.model'
SP_VOCAB_FILE = 'tokenizer.vocab'


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def _validate_text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError(f'encode() expects str, got {type(text).__name__}')
    return text


def _validate_ids(ids: Iterable[Any]) -> list[int]:
    try:
        return [int(token_id) for token_id in ids]
    except (TypeError, ValueError) as exc:
        raise TypeError('decode() expects an iterable of integers') from exc


def _iter_corpus_lines(input_path: Path) -> Iterable[str]:
    if not input_path.exists():
        raise FileNotFoundError(f'Corpus not found: {input_path}')
    with input_path.open(encoding='utf-8') as f:
        for line in f:
            yield line.rstrip('\n')


class BaseTokenizer(ABC):
    tokenizer_type: str

    pad_token: str = PAD_TOKEN
    bos_token: str = BOS_TOKEN
    eos_token: str = EOS_TOKEN
    unk_token: str = UNK_TOKEN

    pad_id: int = PAD_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    unk_id: int = UNK_ID

    skip_unk_on_decode: bool = False

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        pass

    @abstractmethod
    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        pass

    def batch_encode(
        self,
        texts: Iterable[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        encoded = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos)
            for text in texts
        ]
        if not encoded:
            return []

        max_length = max(len(ids) for ids in encoded)
        return [
            ids + [self.pad_id] * (max_length - len(ids))
            for ids in encoded
        ]

    def batch_decode(
        self,
        batch_ids: Iterable[Iterable[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    @abstractmethod
    def save(self, output_dir: str | Path) -> None:
        pass

    def _special_ids_to_skip(self, skip_special_tokens: bool) -> set[int]:
        if not skip_special_tokens:
            return set()
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        if self.skip_unk_on_decode:
            special_ids.add(self.unk_id)
        return special_ids


class CharTokenizer(BaseTokenizer):
    tokenizer_type = 'char'

    def __init__(
        self,
        stoi: dict[str, int],
        *,
        skip_unk_on_decode: bool = False,
        min_freq: int = 1,
        input_corpus: str | None = None,
    ) -> None:
        self.stoi = dict(stoi)
        self.itos = {token_id: token for token, token_id in self.stoi.items()}
        self.skip_unk_on_decode = skip_unk_on_decode
        self.min_freq = min_freq
        self.input_corpus = input_corpus
        self._validate_special_tokens()

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        text = _validate_text(text)
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(char, self.unk_id) for char in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        token_ids = _validate_ids(ids)
        skip_ids = self._special_ids_to_skip(skip_special_tokens)
        chars: list[str] = []
        for token_id in token_ids:
            if token_id in skip_ids:
                continue
            if token_id == self.unk_id:
                chars.append(self.unk_token)
                continue
            token = self.itos.get(token_id)
            if token is None or token in SPECIAL_TOKENS:
                chars.append(self.unk_token)
                continue
            chars.append(token)
        return ''.join(chars)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_path = output_dir / CHAR_TOKENIZER_FILE
        _write_json(
            tokenizer_path,
            {
                'tokenizer_type': self.tokenizer_type,
                'stoi': self.stoi,
            },
        )
        _write_json(output_dir / CONFIG_FILE, self._build_config(str(tokenizer_path.name)))

    @classmethod
    def train(
        cls,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        min_freq: int = 1,
        skip_unk_on_decode: bool = False,
    ) -> CharTokenizer:
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        counter: Counter[str] = Counter()
        for line in _iter_corpus_lines(input_path):
            counter.update(line)

        stoi = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID,
        }
        next_id = len(stoi)
        for char, freq in sorted(counter.items()):
            if char == '\n':
                continue
            if freq < min_freq:
                continue
            stoi[char] = next_id
            next_id += 1

        tokenizer = cls(
            stoi,
            skip_unk_on_decode=skip_unk_on_decode,
            min_freq=min_freq,
            input_corpus=str(input_path),
        )
        tokenizer.save(output_dir)
        return tokenizer

    @classmethod
    def load(cls, tokenizer_dir: str | Path) -> CharTokenizer:
        tokenizer_dir = Path(tokenizer_dir)
        config = _read_json(tokenizer_dir / CONFIG_FILE)
        tokenizer_file = config.get('tokenizer_file', CHAR_TOKENIZER_FILE)
        tokenizer_data = _read_json(tokenizer_dir / tokenizer_file)

        return cls(
            tokenizer_data['stoi'],
            skip_unk_on_decode=config.get('skip_unk_on_decode', False),
            min_freq=config.get('min_freq', 1),
            input_corpus=config.get('input_corpus'),
        )

    def _build_config(self, tokenizer_file: str) -> dict[str, Any]:
        return {
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'unk_token': self.unk_token,
            'pad_id': self.pad_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
            'unk_id': self.unk_id,
            'skip_unk_on_decode': self.skip_unk_on_decode,
            'min_freq': self.min_freq,
            'input_corpus': self.input_corpus,
            'tokenizer_file': tokenizer_file,
        }

    def _validate_special_tokens(self) -> None:
        expected = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID,
        }
        for token, token_id in expected.items():
            if self.stoi.get(token) != token_id:
                raise ValueError(
                    f'CharTokenizer requires {token}={token_id}, got {self.stoi.get(token)}'
                )


class SentencePieceTokenizer(BaseTokenizer):
    tokenizer_type = 'sentencepiece'

    def __init__(
        self,
        model_path: Path,
        *,
        model_type: str,
        skip_unk_on_decode: bool = False,
        normalization_rule_name: str = 'identity',
        character_coverage: float = 1.0,
        input_corpus: str | None = None,
        input_sentence_size: int | None = None,
        shuffle_input_sentence: bool = False,
    ) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                'sentencepiece is required for SentencePieceTokenizer'
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(f'SentencePiece model not found: {model_path}')

        self._processor = spm.SentencePieceProcessor()
        self._processor.load(str(model_path))
        self.model_path = model_path
        self.model_type = model_type
        self.skip_unk_on_decode = skip_unk_on_decode
        self.normalization_rule_name = normalization_rule_name
        self.character_coverage = character_coverage
        self.input_corpus = input_corpus
        self.input_sentence_size = input_sentence_size
        self.shuffle_input_sentence = shuffle_input_sentence
        self._validate_special_token_ids()

    @property
    def vocab_size(self) -> int:
        return self._processor.get_piece_size()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        text = _validate_text(text)
        return self._processor.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        token_ids = _validate_ids(ids)
        if not skip_special_tokens:
            return self._processor.decode(token_ids)

        skip_ids = self._special_ids_to_skip(skip_special_tokens=True)
        filtered_ids = [token_id for token_id in token_ids if token_id not in skip_ids]
        return self._processor.decode(filtered_ids)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dst = output_dir / SP_MODEL_FILE
        if self.model_path.resolve() != model_dst.resolve():
            shutil.copy2(self.model_path, model_dst)

        vocab_src = self.model_path.with_suffix('.vocab')
        vocab_dst = output_dir / SP_VOCAB_FILE
        if vocab_src.exists() and vocab_src.resolve() != vocab_dst.resolve():
            shutil.copy2(vocab_src, vocab_dst)

        _write_json(output_dir / CONFIG_FILE, self._build_config())

    @classmethod
    def train(
        cls,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        model_type: str,
        vocab_size: int,
        skip_unk_on_decode: bool = False,
        normalization_rule_name: str = 'identity',
        character_coverage: float = 1.0,
        input_sentence_size: int | None = None,
        shuffle_input_sentence: bool = True,
    ) -> SentencePieceTokenizer:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                'sentencepiece is required for SentencePieceTokenizer'
            ) from exc

        if model_type not in {'bpe', 'unigram'}:
            raise ValueError(f'Unsupported SentencePiece model_type: {model_type}')
        if vocab_size <= len(SPECIAL_IDS):
            raise ValueError(
                f'vocab_size must be greater than number of special tokens ({len(SPECIAL_IDS)})'
            )

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_prefix = str(output_dir / 'tokenizer')
        train_kwargs: dict[str, Any] = {
            'input': str(input_path),
            'model_prefix': model_prefix,
            'model_type': model_type,
            'vocab_size': vocab_size,
            'pad_id': PAD_ID,
            'bos_id': BOS_ID,
            'eos_id': EOS_ID,
            'unk_id': UNK_ID,
            'pad_piece': PAD_TOKEN,
            'bos_piece': BOS_TOKEN,
            'eos_piece': EOS_TOKEN,
            'unk_piece': UNK_TOKEN,
            'character_coverage': character_coverage,
            'normalization_rule_name': normalization_rule_name,
            'train_extremely_large_corpus': True,
        }
        if input_sentence_size is not None and input_sentence_size > 0:
            train_kwargs['input_sentence_size'] = input_sentence_size
            train_kwargs['shuffle_input_sentence'] = shuffle_input_sentence

        spm.SentencePieceTrainer.train(**train_kwargs)

        tokenizer = cls(
            Path(f'{model_prefix}.model'),
            model_type=model_type,
            skip_unk_on_decode=skip_unk_on_decode,
            normalization_rule_name=normalization_rule_name,
            character_coverage=character_coverage,
            input_corpus=str(input_path),
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=(
                shuffle_input_sentence if input_sentence_size and input_sentence_size > 0 else False
            ),
        )
        tokenizer.save(output_dir)
        return tokenizer

    @classmethod
    def load(cls, tokenizer_dir: str | Path) -> SentencePieceTokenizer:
        tokenizer_dir = Path(tokenizer_dir)
        config = _read_json(tokenizer_dir / CONFIG_FILE)
        model_file = config.get('model_file', SP_MODEL_FILE)

        return cls(
            tokenizer_dir / model_file,
            model_type=config['model_type'],
            skip_unk_on_decode=config.get('skip_unk_on_decode', False),
            normalization_rule_name=config.get('normalization_rule_name', 'identity'),
            character_coverage=config.get('character_coverage', 1.0),
            input_corpus=config.get('input_corpus'),
            input_sentence_size=config.get('input_sentence_size'),
            shuffle_input_sentence=config.get('shuffle_input_sentence', False),
        )

    def _build_config(self) -> dict[str, Any]:
        return {
            'tokenizer_type': self.tokenizer_type,
            'model_type': self.model_type,
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'unk_token': self.unk_token,
            'pad_id': self.pad_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
            'unk_id': self.unk_id,
            'skip_unk_on_decode': self.skip_unk_on_decode,
            'normalization_rule_name': self.normalization_rule_name,
            'character_coverage': self.character_coverage,
            'input_corpus': self.input_corpus,
            'input_sentence_size': self.input_sentence_size,
            'shuffle_input_sentence': self.shuffle_input_sentence,
            'model_file': SP_MODEL_FILE,
            'vocab_file': SP_VOCAB_FILE,
        }

    def _validate_special_token_ids(self) -> None:
        expected = {
            'pad_id': PAD_ID,
            'bos_id': BOS_ID,
            'eos_id': EOS_ID,
            'unk_id': UNK_ID,
        }
        actual = {
            'pad_id': self._processor.pad_id(),
            'bos_id': self._processor.bos_id(),
            'eos_id': self._processor.eos_id(),
            'unk_id': self._processor.unk_id(),
        }
        if actual != expected:
            raise ValueError(
                'SentencePiece special token IDs must be fixed to '
                f'{expected}, got {actual}'
            )


class TokenizerFactory:
    @staticmethod
    def load(tokenizer_dir: str | Path) -> BaseTokenizer:
        tokenizer_dir = Path(tokenizer_dir)
        config_path = tokenizer_dir / CONFIG_FILE
        config = _read_json(config_path)
        tokenizer_type = config['tokenizer_type']

        if tokenizer_type == 'char':
            return CharTokenizer.load(tokenizer_dir)
        if tokenizer_type == 'sentencepiece':
            return SentencePieceTokenizer.load(tokenizer_dir)
        raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')

    @staticmethod
    def train(
        tokenizer_type: str,
        input_path: str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> BaseTokenizer:
        if tokenizer_type == 'char':
            return CharTokenizer.train(
                input_path=input_path,
                output_dir=output_dir,
                min_freq=kwargs.get('min_freq', 1),
                skip_unk_on_decode=kwargs.get('skip_unk_on_decode', False),
            )

        model_type = kwargs.pop('model_type', None)
        if tokenizer_type == 'sentencepiece_bpe':
            model_type = 'bpe'
        elif tokenizer_type == 'sentencepiece_unigram':
            model_type = 'unigram'
        elif tokenizer_type == 'sentencepiece' and model_type is None:
            raise ValueError("model_type is required when tokenizer_type='sentencepiece'")

        if model_type in {'bpe', 'unigram'}:
            vocab_size = kwargs.get('vocab_size')
            if vocab_size is None:
                raise ValueError('vocab_size is required for SentencePiece training')
            return SentencePieceTokenizer.train(
                input_path=input_path,
                output_dir=output_dir,
                model_type=model_type,
                vocab_size=vocab_size,
                skip_unk_on_decode=kwargs.get('skip_unk_on_decode', False),
                normalization_rule_name=kwargs.get('normalization_rule_name', 'identity'),
                character_coverage=kwargs.get('character_coverage', 1.0),
                input_sentence_size=kwargs.get('input_sentence_size'),
                shuffle_input_sentence=kwargs.get('shuffle_input_sentence', True),
            )

        raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')


def load_tokenizer(tokenizer_dir: str | Path) -> BaseTokenizer:
    return TokenizerFactory.load(tokenizer_dir)
