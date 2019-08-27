"""Helpful visualizations of activpal data
"""
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from pandas import isna
from itertools import chain
from datetime import timedelta
from activpal_tools import utils
import seaborn as sns
import logging
logger = logging.getLogger(__name__)

default_mappings = {
    'sitting': 0,
    'standing': 1,
    'lying': 2,
    'sitting_noise': 3,
    'standing_noise': 4,
    'walking': 5,
    'walking_brisk': 6,
    'jogging': 7,
    'running': 8,
    'conditioning': 9,
    "na": -1,
    "sedentary": 0,
    "standing": 1,
    "stepping": 5
}


def map_var(var, mapping=default_mappings):
    if isna(var):
        return -1
    return mapping[var]


def plot_class_state(data, val_col, hue, mapping=default_mappings, linewidth=40, linewidth_scaling=1):
    """Plots the phase-change diagram of the dataset

    Args:
        data (TYPE): Description
        val_col (TYPE): Description
        hue (TYPE): Description
        mapping (None, optional): mapping from value column to encoded real value

    Returns:
        TYPE: Description
    """
    # if no mapping enxists, create a mappin
    lines = []
    for index, row in data.iterrows():
        start, end = mdates.date2num(row["start"]), mdates.date2num(row["end"])
        val = row[val_col]
        # Encode the value into a real number
        encoded = map_var(val, mapping)
        lines.append(((start, encoded), (end, encoded)))
    lc = LineCollection(lines, linewidths=40, colors=hue, alpha=0.5)
    fig, ax = plt.gcf(), plt.gca()

    ax.add_collection(lc)

    monthFmt = mdates.DateFormatter("%H:%M:%S")
    ax.xaxis.set_major_formatter(monthFmt)
    ax.autoscale_view()


def plot_class_states(data, val_cols, hues, mapping=default_mappings, compress_mappings=True):
    """Summary

    Args:
        data (TYPE): Description
        val_cols (TYPE): Description
        hues (TYPE): Description
        mapping (TYPE): Mapping used to encode the information and
            produce yticks. if two labels go to the same encoding, the first label is used.
        subset (TYPE): Description
    """
    # Find out what tagss are present in thsi dataset
    encodings_present = set()
    for val_col in val_cols:
        encodings = data[val_col].apply(lambda x: map_var(x, mapping=mapping))
        encodings = set(encodings)
        encodings_present.update(encodings)
    logger.debug(encodings_present)

    # Compress the mappings for cleaner visualization
    linewidth_scaling = 1
    if compress_mappings:
        new_mapping = {tag_name: encoding for tag_name, encoding in mapping.items() if encoding in encodings_present}
        logger.debug(f"new_mapping: {new_mapping}")
        # Fill up the empty slots caused by missing encodings by mapping them to a minimal set
        cur_encodings = sorted(set(new_mapping.values()))
        encoding_map = {old: new for old, new in zip(cur_encodings, range(-1, len(cur_encodings) - 1))}
        # Modify the scaling
        # print(mapping.values, new_mapping.values)
        linewidth_scaling *= len(mapping.values()) / len(set(new_mapping.values()))
        mapping = {tag_name: encoding_map[encoding] for tag_name, encoding in new_mapping.items()}
        # Update the present_encodings
        # encodings_present = set(encoding_map.values)
        logger.debug(mapping)


    # Plot as necessary
    for val_col, hue in zip(val_cols, hues):
        encodings = plot_class_state(data, val_col, hue,
                                     mapping=mapping, linewidth_scaling=linewidth_scaling)




    # Create reverse mapping from the mapping (priority matters here)
    reverse_mapping = {}
    logger.debug(f"encodings_present: {encodings_present}")
    for key, encoding in mapping.items():
        if encoding in reverse_mapping:
            logger.debug(f"skipping encoding={encoding} with tag={key}")
            continue
        reverse_mapping[encoding] = key
    logger.debug(f"reverse_mapping={reverse_mapping}")
    plt.yticks(list(reverse_mapping.keys()),
               list(reverse_mapping.values()))


def plot_class_states_split(data, val_cols, hues, mapping=default_mappings,
                            fig_params={"figsize": (30, 10)}):
    for trial in set(data.trial):
        print(trial, type(trial))
        if not trial or isna(trial):
            continue
        fig = plt.figure(**fig_params)
        subdata = data.loc[data.trial == trial]
        plot_class_states(subdata, val_cols, hues, mapping)
        plt.title(trial)
        plt.show()


def plot_stepping(data, mapping=default_mappings, resolution=5):
    stepping_df = data.loc[data.trial == "stepping"].copy()
    # might want to migrate these
    stepping_df["METs"] = stepping_df["Activity Score (MET.h)"] * 3600 / stepping_df["duration"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(30, 15))
    plt.sca(ax1)
    plt.yticks(fontsize=24)
    plot_class_states(stepping_df, val_cols=["tag", "activpal_event"], hues=["blue", "orange"])
    xticks = plt.xticks(fontsize=24)
    xlim = plt.xlim()
    # METs
    plt.sca(ax2)
    plt.ylabel("METs")
    plt.yticks(fontsize=24)
    plt.plot(stepping_df.start + stepping_df.duration.apply(lambda x: timedelta(seconds=x)), stepping_df.METs,
             alpha=0.5, linewidth=3)
    plt.xticks(xticks[0], fontsize=24)
    plt.xlim(*xlim)
    plt.title("METs", fontsize=24)

    # Cadence
    sns.set_style("whitegrid")
    plt.sca(ax3)
    plt.yticks(fontsize=24)
    plt.title("Cadence", fontsize=24)
    plt.ylabel("Steps/Minute")
    plt.plot(stepping_df.start + stepping_df.duration.apply(lambda x: timedelta(seconds=x)), stepping_df.step_rate * 60,
             alpha=0.5, linewidth=3)
    # Add in the epoch'd data
    logger.info ("Converting to epochs")
    epoch_df = utils.to_epochs(stepping_df, resolution=resolution)
    plt.plot(epoch_df.start + epoch_df.duration.apply(lambda x: timedelta(seconds=x)), epoch_df.cadence,
             alpha=0.5, linewidth=4, color="black")
    plt.xticks(xticks[0], fontsize=24)
    plt.xlim(*xlim)
    monthFmt = mdates.DateFormatter("%H:%M:%S")
    ax2.xaxis.set_major_formatter(monthFmt)
    ax3.xaxis.set_major_formatter(monthFmt)
    plt.tight_layout()

