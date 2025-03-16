import pandas as pd

def optimize_memory_usage(df):
    """
    Iterates over all columns in the DataFrame and optimizes memory usage 
    by downcasting numerical columns (float64 and int64) to more efficient types.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to optimize.

    Returns:
    pd.DataFrame: The DataFrame with optimized memory usage.
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'float64':
            # Try to downcast float64 to float32 or smaller if possible
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        elif col_type == 'int64':
            # Try to downcast int64 to int32, int16, or int8 if possible
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df