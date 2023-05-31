# lct-2023

## Prepare data
Download data to ```cfg.data_dir```

```python3 prepare_metadata.py --data-dir cfg.data_dir```

## Train

```
for every cfg in ./configs/
python3 train_tok_old.py --config config_{n}.py
```

## Tree structure 

```
/path/to/exp/dir
├── artifacts
│   ├── label.npy
│   ├── test_1.csv
│   ├── test_2.csv
│   ├── test_3.csv
│   ├── test_4.csv
│   ├── train_1.csv
│   ├── train_2.csv
│   ├── train_3.csv
│   ├── train_4.csv
│   ├── val_1.csv
│   ├── val_1.npy
│   ├── val_2.csv
│   ├── val_2.npy
│   ├── val_3.csv
│   ├── val_3.npy
│   ├── val_4.csv
│   └── val_4.npy
└── weights
    ├── model_0.92509.pth
    ├── model_0.92822.pth
    ├── model_0.92835.pth
    └── model_0.93023.pth
```

Take ```test_3.csv``` from each folder, average predicitons, try submitting
