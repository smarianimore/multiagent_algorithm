# Code to transform the dataset in int 0/1 values
import pandas as pd
import random


def evidence_to_numeric(evidence):
    node = str(tuple(evidence.items())[0][0])
    value = tuple(evidence.items())[0][1]

    numericals = ['H', 'C']

    # A conversion is required only when dealing with doable numerical nodes
    if node in numericals:
        if node == 'H' or node == 'C':
            return {node: random.randint(0, 500) if value == 0 else random.randint(501, 1000)}
    else:
        return evidence


def conversion_2(dataset, nodes=None):
    # Avoid warnings
    # pd.options.mode.chained_assignment = None

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


def conversion(dataset, nodes=None):

    df = dataset.copy()

    # Converting the values from Double to Int (Binary)
    # T
    if 'T' in df.columns:
        df.loc[df['T'] <= 298.15, 'T'] = 0
        df.loc[df['T'] > 298.15, 'T'] = 1

    # O
    if 'O' in df.columns:
        df.loc[df['O'] <= 290.15, 'O'] = 0
        df.loc[df['O'] > 290.15, 'O'] = 1

    # H
    if 'H' in df.columns:
        df.loc[df['H'] <= 500, 'H'] = 0
        df.loc[df['H'] > 500, 'H'] = 1

    # C
    if 'C' in df.columns:
        df.loc[df['C'] <= 500, 'C'] = 0
        df.loc[df['C'] > 500, 'C'] = 1

    # Pow
    if 'Pow' in df.columns:
        df.loc[df['Pow'] <= 100000, 'Pow'] = 0
        df.loc[df['Pow'] > 100000, 'Pow'] = 1

    # CO
    if 'CO' in df.columns:
        df.loc[df['CO'] <= 15, 'CO'] = 0
        df.loc[df['CO'] > 15, 'CO'] = 1

    # CO2
    if 'CO2' in df.columns:
        df.loc[df['CO2'] <= 1400, 'CO2'] = 0
        df.loc[df['CO2'] > 1400, 'CO2'] = 1

    df = df.astype(int)

    return df


# if __name__ == '__main__':
#     dataset = pd.read_csv("dataset.csv", sep=',').drop(columns=['timestamp'])[:5]
#     print(dataset)
#
#     # mod1 = conversion(dataset)
#     # print(mod1)
#
#     # mod2 = conversion_2(dataset)
#     # print(mod2)





