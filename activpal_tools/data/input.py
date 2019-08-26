"""Input and normalization of activpal data
takes in activepal data and spits out a coherent, consistent dataset for both epoch
and event-based data
"""
import os
import pandas as pd


def read_epochs(path):
    """Reads epoch information into. refer to http://docs.palt.com/display/EX/15+second+epoch+csv for
    base template.
    Note, epochs may combine events in a single epoch with no temporal separation. Thus, within the epoch,
    Event order will not be accurate. For event temporal order, use the events instead

    Example:
        Epoch 1:

    Args:
        path (TYPE): Description
    """
    pass


def read_events(path, augment=True):
    """Reads the event information into a pandas dataframe
    augments the data with proper steps/interval and rates.

    Args:
        path (TYPE): Description
        augment (bool, optional): Add on the following data fields:
            activity_rate (MET hours)
            steps: (total steps for window, both strides)
            cadence: Steps/minute

    Returns:
        TYPE: Description
    """
    df = pd.read_csv(
        path,
        skiprows=1,
        sep=";",
        index_col=False,
    )
    # correct for time
    df["datetime"] = pd.to_datetime(df["Time(approx)"].values)

    """Mutates the dataset in place, adding necessary data to it"""
    if augment:
        # Add in the activity rate (this is constant aacross the subsegments)
        df["activity_rate"] = df["Activity Score (MET.h)"] / df["Duration (s)"]
        # Convert MET.h to METs
        df["MET"] = df["activity_rate"] * 3600
        # breakpoint()
        df["steps"] = df["Cumulative Step Count"] - pd.concat([pd.Series([0]), df["Cumulative Step Count"].iloc[:-1]])\
                                                      .reset_index(drop=True)
        # Double up on steps as they're single-stried comparisons
        df["steps"] = df["steps"] * 2
        df["step_rate"] = df["steps"] / df["Duration (s)"]
        # Cadence in steps/min
        df["point_cadence"] = df["steps"] / (df["Duration (s)"]) * 60
        # Human readable event types
        event_map = {0: "sedentary", 1: "standing", 2: "stepping",
                     2.1: "stepping", 3.1: "lying_pri", 3.2: "lying_sec", 4: "non-wear", 5: "na"}
        df["activpal_event"] = df["Event Type"].apply(lambda x: event_map[x])

    return df

