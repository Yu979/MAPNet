import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple


def stratified_split(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset with stratification."""
    test_size = config.get('data.split.test_size', 0.2)
    n_bins = config.get('data.split.n_bins', 5)
    random_state = config.get('seed', 42)
    
    # Create stratification labels.
    df = df.copy()
    try:
        df['stratify_bin'] = pd.qcut(df['avg_inhibit'], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Use cut when there is too little data.
        df['stratify_bin'] = pd.cut(df['avg_inhibit'], bins=n_bins, labels=False, duplicates='drop')
    
    # Stratified split.
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['stratify_bin'],
        random_state=random_state
    )
    
    return train_df, val_df



def stratified_kfold_split(df: pd.DataFrame, config, n_splits: int = 5) -> list[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    5-fold stratified cross-validation split.
    
    Args:
        df: complete dataset
        config: configuration object
        n_splits: number of folds, defaults to 5
    
    Returns:
        List of (train_df, val_df) tuples with length n_splits.
        Each fold's validation set is approximately 1/n_splits of the data (20% for 5-fold).
    """
    n_bins = config.get('data.split.n_bins', 5)
    random_state = config.get('seed', 42)
    
    print(f"\n{'='*80}")
    print(f"Creating {n_splits}-fold stratified cross-validation split")
    print(f"{'='*80}")
    
    # Create stratification labels.
    df_copy = df.copy()
    try:
        df_copy['stratify_bin'] = pd.qcut(
            df_copy['avg_inhibit'], 
            q=n_bins, 
            labels=False, 
            duplicates='drop'
        )
    except ValueError:
        # Use cut when there is too little data.
        df_copy['stratify_bin'] = pd.cut(
            df_copy['avg_inhibit'], 
            bins=n_bins, 
            labels=False, 
            duplicates='drop'
        )
    
    print(f"Using {n_bins} stratification bins")
    print(f"Random seed: {random_state}")
    print(f"Total samples: {len(df_copy)}")
    print(f"Validation set per fold: about {len(df_copy)/n_splits:.0f} samples ({100/n_splits:.1f}%)\n")
    
    # Create folds with StratifiedKFold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_copy, df_copy['stratify_bin'])):
        train_df = df_copy.iloc[train_idx].copy()
        val_df = df_copy.iloc[val_idx].copy()
        
        # Remove temporary columns.
        if 'stratify_bin' in train_df.columns:
            train_df = train_df.drop('stratify_bin', axis=1)
        if 'stratify_bin' in val_df.columns:
            val_df = val_df.drop('stratify_bin', axis=1)
        
        folds.append((train_df, val_df))
        
    
    return folds
