import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class CreditCardDataset():
    def __init__(self) -> None:
        self.raw_df = pd.read_csv(
            'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
        self.columns = self.raw_df.columns

    def __prepare_data(self):
        cleaned_df = self.raw_df.copy()

        # You don't want the `Time` column.
        cleaned_df.pop('Time')

        # The `Amount` column covers a huge range. Convert to log-space.
        eps = 0.001  # 0 => 0.1¢
        cleaned_df['Log Amount'] = np.log(cleaned_df.pop('Amount')+eps)

        # Use a utility from sklearn to split and shuffle your dataset.
        train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.2)

        # Form np arrays of labels and features.
        train_labels = np.array(train_df.pop('Class'))
        # bool_train_labels = train_labels != 0
        val_labels = np.array(val_df.pop('Class'))
        test_labels = np.array(test_df.pop('Class'))

        train_features = np.array(train_df)
        val_features = np.array(val_df)
        test_features = np.array(test_df)

        return {
            "train_labels": train_labels,
            "val_labels": val_labels,
            "test_labels": test_labels,
            "train_features": train_features,
            "valid_features": val_features,
            "test_features": test_features
        }
