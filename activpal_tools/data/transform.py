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


if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint
    from activpal_tools.data.processing import PointerDataTagger
    ann = pd.DataFrame([
        {"start": datetime(2000, 1, 1, 1, 1, 5), "end": datetime(2000, 1, 1, 1, 1, 10), "tag": 1, "activpal_event": 1},
        {"start": datetime(2000, 1, 1, 1, 1, 10), "end": datetime(2000, 1, 1, 1, 1, 20), "tag": 2, "activpal_event": 2},
        {"start": datetime(2000, 1, 1, 1, 1, 20), "end": datetime(2000, 1, 1, 1, 1, 30), "tag": 3, "activpal_event": 3},
        {"start": datetime(2000, 1, 1, 1, 1, 30), "end": datetime(2000, 1, 1, 1, 1, 40), "tag": 4, "activpal_event": 4},
    ])
    test_dataset = pd.DataFrame([
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 0)), "Duration (s)": 10, "steps": 15},
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 10)), "Duration (s)": 10, "steps": 20},
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 20)), "Duration (s)": 15, "steps": 10},
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 35)), "Duration (s)": 11, "steps": 30}
    ])
    dt_test = PointerDataTagger(ann)
    tagged_data = dt_test.tag_data(test_dataset)
    pprint(tagged_data)
    epoched_data = to_epochs(tagged_data)
    print(epoched_data.head())
    assert(epoched_data.steps[0] == 4.5)
    assert(epoched_data.steps.iloc[-1] == 30 / 11)
