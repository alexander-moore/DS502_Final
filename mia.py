import pandas as pd
import numpy as np
import sklearn.preprocessing as pre


def main():
    data = pd.read_csv('data_no_bs.csv')
    dummies = pd.get_dummies(data)
    dummies.to_csv('encoded_data.csv')


if __name__ == '__main__':
    main()