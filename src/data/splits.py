"""Train and validation splits.

The same page appears in dozens of photographs, every layout exists in a
Hungarian and an English version typeset identically, and each photographer's
photos share a phone and a room. `train.csv` has two grouping columns:

* `group`: no page and no photographer appears in two groups;
* `group_strict`: coarser, and in addition both language versions of a
  layout are in the same unit. The competition's test set was cut along it.

Split by `group_strict`. Measured on this data with a ResNet-18 at 768 px,
RMSE by validation scheme against the held-out test:

    random 5-fold   0.113
    by group        0.136
    by group_strict 0.143
    held-out test   0.148

The random split is a fifth too good. `group` is still 8 % too good, because
the two language versions of a layout sit in two neighbouring groups and the
network recognises the layout. `group_strict` lands on the test.

TASK 1 is `build_splits`. The test in `tests/test_splits.py` is red until it
is written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def build_splits(df: pd.DataFrame, group_column: str = "group_strict", val_fraction: float = 0.2,
                 seed: int = config.SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
# 1. take the distinct values of `group_column` in ascending order;
    groups = np.sort(df[group_column].unique())
    
    # 2. shuffle them with `np.random.default_rng(seed).shuffle`;
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    
    # 3. move units to the validation side in that order, stopping as soon as
    #    the validation side holds at least `val_fraction` of the rows.
    total_rows = len(df)
    target_val_rows = total_rows * val_fraction
    
    val_groups = set()
    val_row_count = 0
    
    for group in groups:
        val_groups.add(group)
        # Hány sor tartozik ehhez a csoporthoz?
        val_row_count += (df[group_column] == group).sum()
        
        # Ha elértük vagy átléptük a kívánt arányt, megállunk
        if val_row_count >= target_val_rows:
            break
            
    # Pozicionális indexek (0-tól n-ig) előállítása
    all_indices = np.arange(total_rows)
    
    # Kikeressük, mely sorok tartoznak a validációs csoportokba
    is_val = df[group_column].isin(val_groups).to_numpy()
    
    # Szétválasztjuk az indexeket
    val_idx = all_indices[is_val]
    train_idx = all_indices[~is_val]
    
    return train_idx, val_idx
