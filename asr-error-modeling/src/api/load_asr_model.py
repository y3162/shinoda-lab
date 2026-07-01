import torch.nn as nn

from src.api.inference_preprocessor import InferencePreprocessor


def load_asr_model(
    model_name: str,
) -> tuple[nn.Module, InferencePreprocessor]:
    match model_name:
        case 'parakeet-tdt-0.6b-v2':
            from src.api.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2
            model = ParakeetTDT06BV2()
        case 'whisper-large-v3':
            from src.api.whisper_large_v3 import WhisperLargeV3
            model = WhisperLargeV3()
        case _:
            raise NotImplementedError(f'Model {model_name} not implemented')
    return model, model.create_inference_preprocessor()
