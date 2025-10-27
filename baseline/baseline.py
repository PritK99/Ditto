import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import ast
from collections import Counter
from typing import List, Dict, Tuple
import random
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics.text import BLEUScore
from jiwer import wer, cer
import numpy as np

def split_and_save_csv(csv_path: str, base_name: str,
                       train_ratio=0.8, val_ratio=0.1):
    """
    Splits one CSV into train/val/test and saves them.
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:]
    }

    for split_name, split_df in splits.items():
        path = f"{base_name}_{split_name}.csv"
        split_df.to_csv(path, index=False)
        print(f"Saved {len(split_df)} rows → {path}")

class MetricsCalculator:
    """
    Calculate BLEU, CER, and WER metrics for sequence reconstruction.
    """
    def __init__(self, idx_to_token: Dict[int, str], pad_idx: int = 0):
        self.idx_to_token = idx_to_token
        self.pad_idx = pad_idx

        # Initialize BLEU calculator
        self.bleu_calculator = BLEUScore(n_gram=4)

    def indices_to_tokens(self, indices: torch.Tensor) -> List[List[str]]:
        """
        Convert token indices to list of token strings.

        Args:
            indices: (batch_size, seq_len)

        Returns:
            List of token lists for each sequence
        """
        batch_sequences = []
        for seq in indices:
            tokens = []
            for idx in seq:
                idx_val = idx.item()
                if idx_val != self.pad_idx:  # Skip padding
                    tokens.append(self.idx_to_token.get(idx_val, '<UNK>'))
            batch_sequences.append(tokens)
        return batch_sequences

    def tokens_to_string(self, tokens: List[str]) -> str:
        """Join tokens into a single string."""
        return ' '.join(tokens)

    def calculate_bleu(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate BLEU score.

        Args:
            predictions: (batch_size, seq_len) - predicted token indices
            targets: (batch_size, seq_len) - target token indices

        Returns:
            BLEU score (0-1)
        """
        # Convert indices to tokens
        pred_tokens = self.indices_to_tokens(predictions)
        target_tokens = self.indices_to_tokens(targets)

        # Convert to strings for BLEU calculation
        pred_strings = [self.tokens_to_string(tokens) for tokens in pred_tokens]
        target_strings = [[self.tokens_to_string(tokens)] for tokens in target_tokens]

        # Calculate BLEU
        try:
            bleu_score = self.bleu_calculator(pred_strings, target_strings)
            return bleu_score.item()
        except:
            return 0.0

    def calculate_cer(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Character Error Rate (CER).

        Args:
            predictions: (batch_size, seq_len) - predicted token indices
            targets: (batch_size, seq_len) - target token indices

        Returns:
            CER score (lower is better, 0 is perfect)
        """
        # Convert indices to tokens
        pred_tokens = self.indices_to_tokens(predictions)
        target_tokens = self.indices_to_tokens(targets)

        # Convert to strings
        pred_strings = [self.tokens_to_string(tokens) for tokens in pred_tokens]
        target_strings = [self.tokens_to_string(tokens) for tokens in target_tokens]

        # Calculate CER for each pair and average
        try:
            cer_scores = []
            for pred, target in zip(pred_strings, target_strings):
                if len(target) > 0:  # Avoid empty references
                    cer_score = cer(target, pred)
                    cer_scores.append(cer_score)

            return np.mean(cer_scores) if cer_scores else 0.0
        except:
            return 0.0

    def calculate_wer(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Word Error Rate (WER).

        Args:
            predictions: (batch_size, seq_len) - predicted token indices
            targets: (batch_size, seq_len) - target token indices

        Returns:
            WER score (lower is better, 0 is perfect)
        """
        # Convert indices to tokens
        pred_tokens = self.indices_to_tokens(predictions)
        target_tokens = self.indices_to_tokens(targets)

        # Convert to strings
        pred_strings = [self.tokens_to_string(tokens) for tokens in pred_tokens]
        target_strings = [self.tokens_to_string(tokens) for tokens in target_tokens]

        # Calculate WER for each pair and average
        try:
            wer_scores = []
            for pred, target in zip(pred_strings, target_strings):
                if len(target.split()) > 0:  # Avoid empty references
                    wer_score = wer(target, pred)
                    wer_scores.append(wer_score)

            return np.mean(wer_scores) if wer_scores else 0.0
        except:
            return 0.0

    def calculate_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all metrics (BLEU, CER, WER).

        Args:
            predictions: (batch_size, seq_len) - predicted token indices
            targets: (batch_size, seq_len) - target token indices

        Returns:
            Dictionary with all metrics
        """
        return {
            'bleu': self.calculate_bleu(predictions, targets),
            'cer': self.calculate_cer(predictions, targets),
            'wer': self.calculate_wer(predictions, targets)
        }












# Driver code
c_token_path = "./c_token.csv"
cpp_token_path = "./cpp_tokens.csv"

split_and_save_csv(c_token_path, "obf_c_code")
split_and_save_csv(cpp_token_path, "obf_cpp_code")