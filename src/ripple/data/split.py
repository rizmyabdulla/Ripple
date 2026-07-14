"""Speaker-disjoint train/development/test assignment."""

from __future__ import annotations

from collections import defaultdict

from ripple.data.manifest import DatasetSplit


def assign_speaker_splits(
    speaker_ids: list[str],
    *,
    train_ratio: float = 0.9,
    development_ratio: float = 0.05,
    seed: int = 17,
) -> dict[str, DatasetSplit]:
    """Assign each unique speaker to exactly one split."""
    if train_ratio <= 0 or development_ratio < 0:
        raise ValueError("ratios must be non-negative and train_ratio positive")
    if train_ratio + development_ratio >= 1.0:
        raise ValueError("train_ratio + development_ratio must be < 1")
    unique = sorted(set(speaker_ids))
    if not unique:
        raise ValueError("speaker_ids cannot be empty")
    rng = __import__("random").Random(seed)
    order = unique[:]
    rng.shuffle(order)
    n = len(order)
    n_train = max(1, round(n * train_ratio))
    n_dev = round(n * development_ratio)
    if n_train + n_dev >= n and n > 1:
        n_dev = max(0, n - n_train - 1)
    n_train = min(n_train, n - (1 if n > 1 else 0))
    assignment: dict[str, DatasetSplit] = {}
    for index, speaker in enumerate(order):
        if index < n_train:
            assignment[speaker] = DatasetSplit.TRAIN
        elif index < n_train + n_dev:
            assignment[speaker] = DatasetSplit.DEVELOPMENT
        else:
            assignment[speaker] = DatasetSplit.TEST
    # Ensure every requested split that can exist does when n is large enough
    present = set(assignment.values())
    if n >= 3:
        needed = {
            DatasetSplit.TRAIN,
            DatasetSplit.DEVELOPMENT,
            DatasetSplit.TEST,
        }
        missing = needed - present
        if missing:
            by_split: dict[DatasetSplit, list[str]] = defaultdict(list)
            for speaker, split in assignment.items():
                by_split[split].append(speaker)
            donors = by_split[DatasetSplit.TRAIN]
            for split in missing:
                if len(donors) <= 1:
                    break
                moved = donors.pop()
                assignment[moved] = split
    return assignment
