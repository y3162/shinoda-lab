import argparse
from pathlib import Path
from typing import (
    List,
    Tuple,
)
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import duckdb as db
from src.config import (
    SQL_ROOT,
    DEFAULT_SAMPLE_RATE,
    PARQUET_ROOT,
)
from src.utils.print import print_error
from src.api.load_asr_model import load_asr_model
from src.api.inference_preprocessor import InferencePipeline
from src.utils.noise import (
    get_noise_option,
    NoiseGenerator,
)


def get_target_utterances(
    con: db.DuckDBPyConnection,
    noise_config_id: int,
) -> List[Tuple[int, Path, int]]:
    rows = con.execute(
        """
        SELECT u.id, u.audio_path, u.frame_count
        FROM utterances AS u
        WHERE u.id NOT IN (
            SELECT ar.utterance_id
            FROM asr_results AS ar
            WHERE ar.noise_config_id = ?
        )
        ORDER BY u.frame_count DESC;
        """,
        [noise_config_id],
    ).fetchall()
    return [(row[0], Path(row[1]), row[2]) for row in rows]


class NoisyInferenceDataset(Dataset):
    def __init__(
        self,
        noise_config_id: int,
    ):
        con = db.connect(SQL_ROOT, read_only=True)
        self.utterances = get_target_utterances(con, noise_config_id)
        self.frame_counts = [frame_count for _, _, frame_count in self.utterances]
        self.noise_option = get_noise_option(con, noise_config_id)
        self.noise_generator = NoiseGenerator(self.noise_option)
        con.close()

    def __len__(
        self,
    ) -> int:
        return len(self.utterances)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[int, torch.Tensor]:
        utterance_id, audio_path, _ = self.utterances[idx]
        audio, sr = torchaudio.load(audio_path)
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sr, DEFAULT_SAMPLE_RATE)(audio)
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
        elif audio.dim() != 1:
            print_error(f'Audio tensor must have 1 or 2 dimensions, but got {audio.dim()}')
            raise ValueError(f'Audio tensor must have 1 or 2 dimensions, but got {audio.dim()}')

        noisy_audio = self.noise_generator.generate(audio, DEFAULT_SAMPLE_RATE).audio

        return utterance_id, noisy_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_config_id',  type=int,               required=True)
    parser.add_argument('--model_name',       type=str,               required=True)
    parser.add_argument('--batch_size',       type=int, default=None, required=False)
    parser.add_argument('--num_workers',      type=int, default=16,   required=False)
    parser.add_argument('--prefetch_factor',  type=int, default=4,    required=False)
    parser.add_argument('--max_batch_frames', type=int, default=None, required=False)
    args = parser.parse_args()

    NOISE_CONFIG_ID = args.noise_config_id
    MODEL_NAME = args.model_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    asr_model, preprocessor = load_asr_model(MODEL_NAME)
    asr_model.to(device)
    asr_model.eval()
    asr_model.greedy_mode()

    dataset = NoisyInferenceDataset(NOISE_CONFIG_ID)
    dataloader = preprocessor.create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batch_frames=args.max_batch_frames,
        prefetch_factor=args.prefetch_factor,
        use_cuda=device.type == 'cuda',
    )
    pipeline = InferencePipeline(preprocessor, asr_model, device)

    results = []
    with torch.inference_mode():
        for batch, asr_outputs in tqdm(pipeline.run(dataloader), total=len(dataloader)):
            utterance_ids = batch[0]
            for utterance_id, asr_output in zip(utterance_ids, asr_outputs.results):
                results.append({
                    'utterance_id': utterance_id,
                    'noise_config_id': NOISE_CONFIG_ID,
                    'transcript': asr_output[0].text,
                })

    results_df = pd.DataFrame(results)
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(PARQUET_ROOT / f'results_noise_config_id_{NOISE_CONFIG_ID}.parquet')


if __name__ == '__main__':
    main()
