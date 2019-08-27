"""Functions handling manipulation of datasets
"""

from datetime import datetime, timedelta
from collections import defaultdict
import logging
import pandas as pd

from activpal_tools.utils import dt2float

logger = logging.getLogger(__name__)


def to_epochs(event_data, resolution=3, val_col="tag"):
    """Converts event data to epochs. Labels are selected based off highest contribution

    Args:
        event_data (DataFrame): Pandas DataFrame containing at least a start, end column
        resolution (int, optional): Number of seconds per epoch
        val_col (str, optional): Tag to use. SHOULD NOT EVER HAVE TO CHANGE

    Returns:
        TYPE: Description
    """
    if val_col != "tag":
        logger.warning("Non-tag val_col should generally not be used. Epoch conversions is last step."
                       "Are you sure you wish to continue?")
    # First indentify allocations. Windows should be split and
    # merged using the individual classes. Long-form data
    # holder with contain index start end dur tutples
    holder = [[]]
    # i = 1
    for num, (index, row) in enumerate(event_data.iterrows()):
        if num % 100 == 0:
            print(f"Handling row {num}")
        s, e = row.start, row.end
        while s < e:
            # print(s, e)
            # i += 1
            # if i > 1000:
            #     break
            cur = holder[-1]
            # Determine how much time is in the current row
            if not cur:
                cur_time = 0
            else:
                cur_time = sum(i[-1] for i in cur)
            # Amount of time that needs to be added
            to_find = resolution - cur_time
            # print(s, e, cur_time, to_find)
            # If the instance is full,
            if to_find <= 10e-8:
                # print("APPEND")
                holder.append([])
                continue

            # Add that much time to the current holder from the current event if possible
            overlap = min(e, s + timedelta(seconds=to_find))
            # print(overlap)
            cur.append((index, s, overlap, (overlap - s).total_seconds()))
            s = overlap
    # Then reconstruct a dataframe from such (Breakdown-> compile)
    print(f"Reconstructing holder of length {len(holder)}")
    rows = []
    for interval_num, intervals in enumerate(holder):
        # window here is thus the start to end
        window = (intervals[0][1], intervals[-1][2])
        # Calculate the highest tag for the window
        activpal_event_dist = defaultdict(int)
        tag_dist = defaultdict(int)
        # Also the total steps for the epoch in the meantime
        epoch_steps = 0
        for index, start, end, duration in intervals:
            row = event_data.loc[index, :]
            activpal_event_dist[row["activpal_event"]] += duration
            tag_dist[row[val_col]] += duration
            # Partial steps
            # print(row)
            epoch_steps += row["steps"] * (end-start).total_seconds() / row["duration"]

        event = max(activpal_event_dist.items(), key=lambda x: x[1])[0]
        tag = max(tag_dist.items(), key=lambda x: x[1])[0]
        logger.debug(f"tag_dist{tag_dist}, selected_tag={tag}")
        epoch_start = intervals[0][1]
        epoch_end = intervals[-1][2]
        epoch_duration = (epoch_end - epoch_start).total_seconds()
        data = {
            val_col: tag,
            "activpal_event": event,
            "steps": epoch_steps,
            "cadence": epoch_steps / epoch_duration * 60,
            "start": epoch_start,
            "end": epoch_end,
            "duration": epoch_duration,
        }
        rows.append(data)

    return pd.DataFrame(rows)


def find_transitions(dataset, val_col="tag"):
    """Indentifies all transitions

    Args:
        dataset (TYPE): Description
        val_col (str, optional): Column to detect transtions on

    Returns:
        list of timestamps indicating the transition point
    """
    transitions = []
    cur = None
    for index, row in dataset.iterrows():
        prev, cur = cur, row[val_col]
        # Skip the first case
        if not prev:
            continue
        if cur != prev:
            transitions.append(row["start"])
    logger.info(f"Identified {len(transitions)} transitions")
    return transitions


def trim_transitions(dataset, how="mid", n_seconds=1):
    """Trims the dataset of transition data. Note: Needs to handle breaks, windows of size < n_seconds
    Transitions refer to a transition between True states (annotated).

    Args:
        dataset (TYPE): Description
        how (str, optional): mid, left, right. Sets the trimming boundary
            mid: Splits n_seconds across both boundaries and removes accordingly
            left: removes n_seconds of data from the right edge of the preceding window
            right: removes n_seconds of data from the left edge of the next window
        n_seconds (int, optional): Number of seconds to trip between the t
    """
    logger.info(f"Trimming transitions from a dataset of shape: {dataset.shape}")
    dataset = dataset.copy()
    transitions = find_transitions(dataset)
    transition_p = 0
    to_remove = []
    # trim the left edge
    left_trim = []
    right_trim = []
    for index, row in dataset.iterrows():
        cur_transition = transitions[transition_p]
        start, end = row["start"], row["end"]
        # If the transition is out of scope, goto next
        if start > cur_transition and end > cur_transition:
            transition_p += 1
            # And then do the boundary check
            if transition_p >= len(transitions):
                break
            cur_transition = transitions[transition_p]

        # Specify the bounds
        if how == "mid":
            lbound = cur_transition - timedelta(seconds=n_seconds / 2)
            rbound = cur_transition + timedelta(seconds=n_seconds / 2)
        elif how == "left":
            lbound = cur_transition - timedelta(seconds=n_seconds)
            rbound = cur_transition
        elif how == "right":
            lbound = cur_transition
            rbound = cur_transition + timedelta(seconds=n_seconds)

        logger.debug(f"Bounds = {(lbound, rbound)}")
        logger.debug(f"s={start}, e={end}, l={lbound}, r={rbound}")

        # Delete the interval if cleanly contained
        if lbound <= start < end <= rbound:
            to_remove.append(index)
        # The right side
        if rbound >= end >= lbound:
            logger.debug("rtrim")
            right_trim.append((index, lbound))
        # Handle the case of trimming the left side
        if lbound <= start <= rbound:
            logger.debug("ltrim")
            left_trim.append((index, rbound))

    logger.info(f"Handling: left_trim={len(left_trim)}, right_trim={len(right_trim)}, to_remove={len(to_remove)}")
    logger.debug(f"left_trim={left_trim}")
    logger.debug(f"right_trim={right_trim}")
    logger.debug(f"to_remove={to_remove}")

    # Trim left
    if left_trim:
        indices, values = zip(*left_trim)
        # dataset.loc[indicies, "start"] = values
        df2 = pd.DataFrame({"start": values}, index=indices)
        dataset.update(df2)

    # Trim rights
    if right_trim:
        indices, values = zip(*right_trim)
        # dataset.loc[indicies, "end"] = values
        df2 = pd.DataFrame({"end": values}, index=indices)
        dataset.update(df2)

    # Drop the unwanted ones
    dataset.drop(to_remove, inplace=True)
    return dataset


if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint
    from activpal_tools.data.processing import PointerDataTagger
    import sys
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # ann = pd.DataFrame([
    #     {"start": datetime(2000, 1, 1, 1, 1, 5), "end": datetime(2000, 1, 1, 1, 1, 10), "tag": 1, "activpal_event": 1},
    #     {"start": datetime(2000, 1, 1, 1, 1, 10), "end": datetime(2000, 1, 1, 1, 1, 20), "tag": 2, "activpal_event": 2},
    #     {"start": datetime(2000, 1, 1, 1, 1, 20), "end": datetime(2000, 1, 1, 1, 1, 30), "tag": 3, "activpal_event": 3},
    #     {"start": datetime(2000, 1, 1, 1, 1, 30), "end": datetime(2000, 1, 1, 1, 1, 40), "tag": 4, "activpal_event": 4},
    # ])
    # test_dataset = pd.DataFrame([
    #     {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 0)), "Duration (s)": 10, "steps": 15},
    #     {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 10)), "Duration (s)": 10, "steps": 20},
    #     {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 20)), "Duration (s)": 15, "steps": 10},
    #     {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 35)), "Duration (s)": 11, "steps": 30}
    # ])
    # dt_test = PointerDataTagger(ann)
    # tagged_data = dt_test.tag_data(test_dataset)
    # pprint(tagged_data)
    # epoched_data = to_epochs(tagged_data)
    # print(epoched_data.head())
    # assert(epoched_data.steps[0] == 4.5)
    # assert(epoched_data.steps.iloc[-1] == 30 / 11)

    # Transition test
    """Dataset generation
    """
    ann = pd.DataFrame([
        {"start": datetime(2000, 1, 1, 1, 1, 5), "end": datetime(2000, 1, 1, 1, 1, 10), "tag": 1, "activpal_event": 1},
        {"start": datetime(2000, 1, 1, 1, 1, 10), "end": datetime(2000, 1, 1, 1, 1, 12), "tag": 2, "activpal_event": 2},
        # Fully contained and should not be trimmed
        {"start": datetime(2000, 1, 1, 1, 1, 12), "end": datetime(2000, 1, 1, 1, 1, 16), "tag": 2, "activpal_event": 2},
        {"start": datetime(2000, 1, 1, 1, 1, 16), "end": datetime(2000, 1, 1, 1, 1, 20), "tag": 2, "activpal_event": 2},
        {"start": datetime(2000, 1, 1, 1, 1, 20), "end": datetime(2000, 1, 1, 1, 1, 30), "tag": 3, "activpal_event": 3},
        {"start": datetime(2000, 1, 1, 1, 1, 30), "end": datetime(2000, 1, 1, 1, 1, 30, 500000), "tag": 4,
         "activpal_event": 4},
        {"start": datetime(2000, 1, 1, 1, 1, 30, 500000), "end": datetime(2000, 1, 1, 1, 1, 34),
         "tag": 4,"activpal_event": 4}
    ])
    print(ann)
    # dt_test = PointerDataTagger(ann)
    # tagged_data = dt_test.tag_data(test_dataset)

    # now lets trim it
    trimmed = trim_transitions(ann, how="mid")
    print(trimmed)
    # First start is not trimmed for now
    # first end should be trimmed
    assert(trimmed.iloc[0]["end"] == datetime(2000, 1, 1, 1, 1, 9, 500000))
    # Second should have only it's start trimmed
    assert(trimmed.iloc[1]["start"] == datetime(2000, 1, 1, 1, 1, 10, 500000))
    assert(trimmed.iloc[1]["end"] == datetime(2000, 1, 1, 1, 1, 12))
    # Third should be untouched
    assert(trimmed.iloc[2]["start"] == datetime(2000, 1, 1, 1, 1, 12))
    assert(trimmed.iloc[2]["end"] == datetime(2000, 1, 1, 1, 1, 16))
    assert 5 not in trimmed.index
