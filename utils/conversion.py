# Code to transform the dataset in int 0/1 values
import pandas as pd


def conversion(dataset, nodes=None):
    # Avoid warnings
    pd.options.mode.chained_assignment = None

    df = dataset.copy()

    # Converting the values from Double to Int (Binary)
    # T
    if 'T' in df.columns:
        df['T'][df['T'] <= 298.15] = 0
        df['T'][df['T'] > 298.15] = 1

    # O
    if 'O' in df.columns:
        df['O'][df['O'] <= 290.15] = 0
        df['O'][df['O'] > 290.15] = 1

    # H
    if 'H' in df.columns:
        df['H'][df['H'] <= 500] = 0
        df['H'][df['H'] > 500] = 1

    # C
    if 'C' in df.columns:
        df['C'][df['C'] <= 500] = 0
        df['C'][df['C'] > 500] = 1

    # Pow
    if 'Pow' in df.columns:
        df['Pow'][df['Pow'] <= 100000] = 0
        df['Pow'][df['Pow'] > 100000] = 1

    # CO
    if 'CO' in df.columns:
        df['CO'][df['CO'] <= 15] = 0
        df['CO'][df['CO'] > 15] = 1

    # CO2
    if 'CO2' in df.columns:
        df['CO2'][df['CO2'] <= 1400] = 0
        df['CO2'][df['CO2'] > 1400] = 1

    df = df.astype(int)

    return df
