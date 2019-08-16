import os
import pandas as pd

def read_events(path):
    """Reads the event information into a pandas dataframe
    augments the data with proper steps/interval and rates
    """
    df = pd.read_csv(
        path,
        skiprows=1,
        sep=";",
        index_col=False,
    )
    """Mutates the dataset in place, adding necessary data to it"""
    # Add in the activity rate (this is constant aacross the subsegments)
    df["activity_rate"] = df["Activity Score (MET.h)"] / df["Duration (s)"]
    # breakpoint()
    df["steps"] = df["Cumulative Step Count"] - pd.concat([pd.Series([0]), df["Cumulative Step Count"].iloc[:-1]])\
                                                  .reset_index(drop=True)

    return df
