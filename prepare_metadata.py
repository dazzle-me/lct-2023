import os
import os.path as osp
import argparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold ## group such items don't overlap?

def create_features(pairs_df: pd.DataFrame, features_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    features = (
        pairs_df
        .merge(
            features_df
            .add_suffix('1'),
            on="variantid1"
        )
        .merge(
            features_df
            .add_suffix('2'),
            on="variantid2"
        )
    )
    if is_train:
        kf = StratifiedGroupKFold(n_splits=5)
        for fold, (_, val_index) in enumerate(kf.split(features, features.target, groups=features.variantid1)):
            features.loc[val_index, 'fold'] = fold
            print(f"fold : {fold}, amount : {len(val_index)}")
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)

    args = parser.parse_args()

    data_dir = args.data_dir
    
    train_df = pd.read_parquet(osp.join(data_dir, "train_data.parquet"))
    train_pairs = pd.read_parquet(osp.join(data_dir, "train_pairs.parquet"))

    test_df = pd.read_parquet(osp.join(data_dir, "test_data.parquet"))
    test_pairs = pd.read_parquet(osp.join(data_dir, "test_pairs_wo_target.parquet"))

    ## create pairs df for inference, for pseudo on train use separate class that 
    ## uses original train_data.parquet file
    train_features = create_features(train_pairs, train_df, is_train=True)
    train_features.to_parquet(osp.join(data_dir, "train_df.parquet"))
    print(train_features.shape, train_df.shape, train_pairs.shape)

    test_features = create_features(test_pairs, test_df, is_train=False)
    test_features['target'] = -1
    test_features.to_parquet(osp.join(data_dir, "test_df.parquet"))
    print(test_features.shape, test_df.shape, test_pairs.shape)

    
     