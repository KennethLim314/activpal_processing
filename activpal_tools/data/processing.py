"""
Module handling initial processing of the data
"""
from datetime import datetime, timedelta
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class PointerDataTagger:
    """Handles the tagging of the activepal data using a reference annotation
    """

    def __init__(self, ann):
        """Summary

        Args:
            ann (pd.Dataframe): Annotation data with start, end columns.
                All other columns are assumed to be related annotations
        """
        self.mapping = {}
        self.keys = []
        # Hardcode the additional columns for now
        ann_data_cols = [i for i in ann.columns if i not in {"start", "end"}]
        # Check and make sure that reserved colnames don't exist and required ones do <NOT IMPLEMENTED>
        for index, row in ann.iterrows():
            start, end = row["start"], row["end"]
            ann_datum = row[ann_data_cols]
            self.mapping[(start, end)] = ann_datum
            self.keys.append((start, end))
        # IMPLEMENT CHECKS ON THE DATASET. CHECK FOR OVERLAPS within the dataset /annotations itself

    def breakdown_dataset(self, dataset, limit=0):
        """Breaks down the dataset into the various distinct windows
        Args:
            dataset (pd.DataFrame): DataFrame containing dataset info. In
                particular, must have the datetime and Duration (s) columns
        returns:
            List of tuples of form
                Index of the dataset row associated with the annotation
                Start (Datetime): Start of the window
                End (Datetime):  End of the interval
                annotations (dict): Object Containing annotation parameters (new columns) for that end
        """
        # Prepare the dataset
        holder = []

        indata = []
        for index, row in dataset.iterrows():
            # Start of activepal is calculated using excel float formatted time
            # https://support.microsoft.com/en-us/help/210276/how-to-store-calculate-and-compare-date-time-data-in-microsoft-access
            start = datetime(1899, 12, 30) + timedelta(days=row["Time"])
            end = start + timedelta(seconds=row["Duration (s)"])
            indata.append((index, start, end))
        # print(indata)

        # First we figure out how to split the original dataset into it's various actual tags such that
        # i.e: True = 0-1 walking, 1-2 running
        # activepal = 0-1.5 walking, 1.5-2 running
        # will create
        # time  | activepal | Real
        # -------------------
        # 0-1   | walking   | walking
        # 1-1.5 | walking   | running
        # 1.5-2 | running   | running
        # We'll do this Two pointer crawl acros the ordered keys and ordered activpal data
        # Then we'll reduce after
        # Pointer to the mapping
        p_tag = 0
        # Current lower and upper bounds for the current tag
        tag_lower, tag_upper = self.keys[0]
        p_data = 0
        index, data_lower, data_upper = indata[0]
        logger.debug(indata)
        # While there is input data to handle
        counter = 0
        breakout = 0
        while True:
            logger.debug(f"Cur={(data_lower.time(), data_upper.time())}, "
                         f"tag_window={(tag_lower.time(), tag_upper.time())} "
                         f"||| cur_equals: {data_lower.time() == data_upper.time()}")
            # handle the case where a window shift is needed
            # Debug limits to break from loops
            counter += 1
            if limit and counter > limit:
                raise
            # Problem seems to be here. Breaks in teh case of multiple tag windows before.
            # Need to handle multiple data windows too
            # For the tagging window
            while tag_lower >= tag_upper or tag_upper < data_lower:
                p_tag += 1
                if p_tag >= len(self.keys):
                    logger.debug("overshot mapping")
                    holder.append((index, data_lower, data_upper, None))
                    breakout = 1
                    break
                tag_lower, tag_upper = self.keys[p_tag]
                logger.debug(f"shifting tag window to {(tag_lower, tag_upper)}")
            if breakout:
                break
            # For the current window
            if data_lower >= data_upper:
                p_data += 1
                if p_data >= len(indata):
                    logger.debug("overshot indata")
                    break
                index, data_lower, data_upper = indata[p_data]
            logger.debug(f"POST_SHIFT: Cur={(data_lower.time(), data_upper.time())}, "
                         f"tag_window={(tag_lower.time(), tag_upper.time())} "
                         f"||| cur_equals: {data_lower.time() == data_upper.time()}\n")

            """Handle cases like
            tag_window :    |----|
            data window: |------|
            OR
            tag_window :       |----|
            data window: |--|
            """
            if data_lower < tag_lower:
                logger.debug(f"FLAG 1")
                # index of the relevant row, start, end, and the tag
                # Find the maximum value of the overlap
                max_overlap = min(data_upper, tag_lower)
                to_append = (index, data_lower, max_overlap, None)
                holder.append(to_append)
                logger.debug(f"APPENDING: {to_append}")
                data_lower = max_overlap

            # tag_window :    |----|
            # data window:        |------|
            # BECOMES
            # tag_window :         |
            # data window:         |-----|
            #                     || -> logged

            # OR
            # tag_window :    |----|
            # data window:      |-|
            # BECOMES
            # tag_window :        ||
            # data window:        |
            #                   |-| -> Logged
            # OR
            # tag_window :    |----|-----|
            # data window:      |-------|
            # BECOMES
            # tag_window :         |-----|
            # data window:         |----|
            #                   |--| -> Logged
            # THEN                 |----| -> Logged
            # tag_window :         |-----|
            # data window:                  |----|
            #                   NOLOG
            elif data_lower >= tag_lower:
                logger.debug(f"FLAG 2")
                # Calculate the furthest overlap
                furthest_overlap = min(data_upper, tag_upper)
                ann_datum = self.mapping[self.keys[p_tag]]
                to_append = (index, data_lower, furthest_overlap, ann_datum.to_dict())
                tag_lower = furthest_overlap
                data_lower = furthest_overlap
                logger.debug(f"Overlap found at {furthest_overlap}, ann_datum={ann_datum}")
                # Check if there's an actual overlap to
                logger.debug(f"APPENDING: {to_append}")
                holder.append(to_append)
        return holder

    def compile_data(self, breakdown, dataset):
        """Uses the original dataset data and the breakdown and mashes them together accounting for scaling
        of the various values (splitting of attributes across segments). Hardcoded for now

        Args:
            breakdown (TYPE): Description
            dataset (TYPE): Description
        Returns:
            DataFrame containing
                start
                end
                activepal_dat
                true_tag
                (all other dataset tags)
        """
        # non numeric columns in activpal
        non_numeric = {
            "Data Count",
            "activity_rate",  # (Rates do not need to be distributed)
            "Event Type",  # Encoded as integer by activepal
            "activpal_event",
            "trial",
        }
        # non_numeric = {
        #     "AbsSumDiffX",
        #     "AbsSumDiffY",
        #     "AbsSumDiffZ",
        # }

        blacklist = {
            "Time",
            "Time(approx)",
            "datetime",
            "Duration (s)",
            "Cumulative Step Count",
            "Waking Day"
        }
        # Columns that we don't want
        result = []
        for index, start, end, ann_datum in breakdown:
            # Find the row that's the basis for the segment
            row = dataset.loc[index]
            logger.debug(f"row: {row}")
            segment_data = {}
            segment_duration = (end - start).total_seconds()
            logger.debug(f"breakdown data: {start}, {end}, ann_datum={ann_datum}. Duration={segment_duration}")
            # How much of the whole dataset is the segment
            segment_prop = segment_duration / row["Duration (s)"]
            try:
                assert(segment_prop <= 1)
            except AssertionError:
                print(f"Segment prop of value={segment_prop} > 1")
                print(f"index={index}, start={start}, end={end}, ann_datum={ann_datum.to_dict()}")
                print(f"row={row.to_dict()}")
            # Determine which rows to keep/normalize
            for key, item in row.to_dict().items():
                # print(key)
                if key in blacklist:
                    continue
                if key in non_numeric:
                    segment_data[key] = item
                # if numeric, we need to normalize
                else:
                    # logger.debug(f"Normalizing {key}")
                    try:
                        segment_data[key] = item * segment_prop
                    except:
                        logger.error(f"Unable to proportionize key={key} with value={item}")
                        raise
            # raise

            # Now we add in all the useful data

            segment_data["start"] = start
            segment_data["end"] = end
            segment_data["duration"] = (end - start).total_seconds()
            if ann_datum:
                for key, item in ann_datum.items():
                    segment_data[key] = item
                # segment_data["trial"] = trial
                # segment_data["trial_set"] = trial_set
            result.append(segment_data)
        df = pd.DataFrame(result)
        return df

    def tag_data(self, dataset):
        """Integration method

        Args:
            dataset (TYPE): Description
        Returns:
            DataFrame containing
                start
                end
                activepal_dat
                true_tag
                (all other dataset tags)
        """
        breakdown = self.breakdown_dataset(dataset)
        logger.debug(f"Breakdown: {breakdown}")
        compiled = self.compile_data(breakdown, dataset)
        return compiled



if __name__ == "__main__":
    from datetime import datetime
    import sys
    from activpal_tools.utils import dt2float
    from pprint import pprint
    # logging.getLogger().setLevel(logging.DEBUG)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Test dataset
    """Dataset looks like this
    ann:                  1     2     3      4
                        |---|------|------|------|
    Data:            |------|------|---------|------|
                     0      10     20        35     45
    """
    ann = pd.DataFrame([
        {"start": datetime(2000, 1, 1, 1, 1, 5), "end": datetime(2000, 1, 1, 1, 1, 10), "tag_details": 1},
        {"start": datetime(2000, 1, 1, 1, 1, 10), "end": datetime(2000, 1, 1, 1, 1, 20), "tag_details": 2},
        {"start": datetime(2000, 1, 1, 1, 1, 20), "end": datetime(2000, 1, 1, 1, 1, 30), "tag_details": 3},
        {"start": datetime(2000, 1, 1, 1, 1, 30), "end": datetime(2000, 1, 1, 1, 1, 40), "tag_details": 4},
    ])
    test_dataset = pd.DataFrame([
        # This is missing stuff on the left but the right fits in neatly
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 0)), "Duration (s)": 10},
        # Fully contained
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 10)), "Duration (s)": 10},
        # Runoff on the right crosses two tag fields
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 20)), "Duration (s)": 15},
        # Runoff on the right into the void
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 35)), "Duration (s)": 10}
    ])
    dt_test = PointerDataTagger(ann)
    dt_breakdown = dt_test.breakdown_dataset(test_dataset)
    pprint(dt_breakdown)
    # Unit tests for the breakdown
    assert(len(dt_breakdown) == 7)
    assert([i[-1]["tag_details"] if i[-1] else i[-1] for i in dt_breakdown] == [None, 1, 2, 3, 4, 4, None])
    # TEST THE DURATION (NOT IMPLEMENTED)
    # assert
    # Tests datasets completely outside the annotations
    ann = pd.DataFrame([
        {"start": datetime(2000, 1, 1, 1, 1, 15), "end": datetime(2000, 1, 1, 1, 1, 20), "tag_details": 1},
        {"start": datetime(2000, 1, 1, 1, 1, 20), "end": datetime(2000, 1, 1, 1, 1, 30), "tag_details": 2},
    ])
    test_dataset = pd.DataFrame([
        # This is completely removed
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 0)), "Duration (s)": 10},
        # This is missing stuff on the left but the right fits in neatly
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 10)), "Duration (s)": 10},
    ])
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # THis dataset tests gapped annotations
    dt_test = PointerDataTagger(ann)
    dt_breakdown = dt_test.breakdown_dataset(test_dataset)
    # THe first window should be 0-10s
    print(dt_breakdown)
    try:
        assert(dt_breakdown[0][1].second == 0 and dt_breakdown[0][2].second == 10)
    except:
        raise
    # One more test to see what multiple tag windows before the data looks like
    """Dataset looks like this
    ann:                  1     2     3      4
                        |---|------|------|------|
    Data:                            |------|
                     0      10     20        35     45
    """
    ann = pd.DataFrame([
        {"start": datetime(2000, 1, 1, 1, 1, 5), "end": datetime(2000, 1, 1, 1, 1, 10), "tag_details": 1},
        {"start": datetime(2000, 1, 1, 1, 1, 10), "end": datetime(2000, 1, 1, 1, 1, 20), "tag_details": 2},
        {"start": datetime(2000, 1, 1, 1, 1, 20), "end": datetime(2000, 1, 1, 1, 1, 30), "tag_details": 3},
        {"start": datetime(2000, 1, 1, 1, 1, 30), "end": datetime(2000, 1, 1, 1, 1, 40), "tag_details": 4},
    ])
    test_dataset = pd.DataFrame([
        # Preceded by two tag windows
        {"Time": dt2float(datetime(2000, 1, 1, 1, 1, 21)), "Duration (s)": 14},
    ])
    dt_test = PointerDataTagger(ann)
    dt_breakdown = dt_test.breakdown_dataset(test_dataset)
    print(dt_breakdown)
    assert(dt_breakdown[0][1] < dt_breakdown[0][2])
