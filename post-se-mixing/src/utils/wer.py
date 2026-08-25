from typing import Tuple

import jiwer


NORMALIZER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
])


def norm(text: str) -> str:
    return NORMALIZER(text if text is not None else '')


def utterance_errors(ref: str, hyp: str) -> Tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    if r == '':
        return 0, 0
    r_words = r.split()
    measures = jiwer.process_words(r, h)
    return int(measures.substitutions + measures.deletions + measures.insertions), len(r_words)
