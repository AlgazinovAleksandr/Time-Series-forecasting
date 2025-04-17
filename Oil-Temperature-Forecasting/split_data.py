import pandas as pd

def split_dataframe(df, time_column='date'):
    """
    Splits the dataframe into training, validation, and test sets in a 3:1:1 ratio based on time order.
    """
    # Sort the dataframe by the time column
    df = df.sort_values(by=time_column)

    # Calculate the split indices
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # Split the dataframe
    train = df.iloc[:train_end]
    validation = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, validation, test