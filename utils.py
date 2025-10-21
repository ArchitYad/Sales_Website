# utils.py
def parse_mixed_dates(df, col="Date"):
    import pandas as pd
    dates = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    
    mask = dates.isna()
    if mask.any():
        dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format="%m/%d/%Y", errors="coerce")
    mask = dates.isna()
    if mask.any():
        dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format="%d/%m/%Y", errors="coerce")
    mask = dates.isna()
    if mask.any():
        dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format="%m-%d-%Y", errors="coerce")
    mask = dates.isna()
    if mask.any():
        dates.loc[mask] = pd.to_datetime(df.loc[mask, col], format="%d-%m-%Y", errors="coerce")
    
    return dates
