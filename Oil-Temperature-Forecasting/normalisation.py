import pandas as pd
def standardize_data(train, val, test, ignore_columns=['date']):
    """
    Standardizes each feature (zero mean, unit variance) based on training data statistics.
    Applies the same normalization to validation and test sets, ignoring specified columns.
    """
    # Separate the columns to standardize
    train_features = train.drop(columns=ignore_columns)
    val_features = val.drop(columns=ignore_columns)
    test_features = test.drop(columns=ignore_columns)

    # Calculate mean and standard deviation from the training data
    mean = train_features.mean()
    std = train_features.std()

    # Standardize the datasets
    train_normalized = (train_features - mean) / std
    val_normalized = (val_features - mean) / std
    test_normalized = (test_features - mean) / std

    # Add back the ignored columns
    train_normalized[ignore_columns] = train[ignore_columns]
    val_normalized[ignore_columns] = val[ignore_columns]
    test_normalized[ignore_columns] = test[ignore_columns]

    return train_normalized, val_normalized, test_normalized