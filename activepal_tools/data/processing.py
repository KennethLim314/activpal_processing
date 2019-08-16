from datetime import datetime, timedelta
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class PointerDataTagger:
    """Handles the tagging of the activepal
    """
    def __init__(self, ann):
        self.mapping = {}
        self.keys = []
        for start, end, tag_detail in zip(ann.start, ann.end, ann.tag_details):
            self.mapping[(start, end)] = tag_detail
            self.keys.append((start, end))
        # IMPLEMENT CHECKS ON THE DATASET. CHECK FOR OVERLAPS

    def supplement_dataset(self, dataset):
        """Mutates the dataset in place, adding necessary data to it"""
        # Add in the activity rate (this is constant aacross the subsegments)
        dataset["activity_rate"] = dataset["Activity Score (MET.h)"] / dataset["Duration (s)"]
        # breakpoint()
        dataset["steps"] = dataset["Cumulative Step Count"] - pd.concat([pd.Series([0]), dataset["Cumulative Step Count"].iloc[:-1]])\
                                                                .reset_index(drop=True)
        return dataset

    def breakdown_dataset(self, dataset, limit=0):
        """Breaks down the dataset into the various distinct windows
        Args:
            dataset (pd.DataFrame): DataFrame containing dataset info. In
                particular, must have the datetime and Duration (s) columns
        returns:
        List of tuples of form
            Index of the row
            Start
            End
            Tag
        """
        # Prepare the dataset
        holder = []

        indata = []
        for index, row in dataset.iterrows():
            start, end = row["datetime"], row["datetime"] + timedelta(seconds=row["Duration (s)"])
            indata.append((index, start, end))

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
        p_mapping = 0
        # Current lower and upper bounds for the current tag
        map_lower, map_upper = self.keys[0]
        p_in = 0
        index, cur_lower, cur_upper = indata[0]
        logger.debug(indata)
        # While there is input data to handle
        counter = 0
        while True:
            logger.debug(f"Cur={(cur_lower.time(), cur_upper.time())}, "
                         f"tag_window={(map_lower.time(), map_upper.time())} "
                         f"||| cur_equals: {cur_lower.time() == cur_upper.time()}")
            # handle the case where a window shift is needed
            # Debug limits to break from loops
            counter += 1
            if limit and counter > limit:
                raise
            # For the tagging window
            if map_lower >= map_upper:
                p_mapping += 1
                if p_mapping >= len(self.keys):
                    logger.debug("overshot mapping")
                    holder.append((index, cur_lower, cur_upper, None))
                    break
                map_lower, map_upper = self.keys[p_mapping]
                logger.debug(f"shifting tag window to {(map_lower, map_upper)}")
            # For the current window
            if cur_lower >= cur_upper:
                p_in += 1
                if p_in >= len(indata):
                    logger.debug("overshot indata")
                    break
                index, cur_lower, cur_upper = indata[p_in]
            logger.debug(f"POST_SHIFT: Cur={(cur_lower.time(), cur_upper.time())}, "
                         f"tag_window={(map_lower.time(), map_upper.time())} "
                         f"||| cur_equals: {cur_lower.time() == cur_upper.time()}")

            """Handle cases like
            tag_window :    |----|
            data window: |------|
            OR
            tag_window :       |----|
            data window: |--|

            """
            if cur_lower < map_lower:
                logger.debug(f"FLAG 1")
                # index of the relevant row, start, end, and the tag
                # Find the maximum value of the overlap
                max_overlap = min(cur_upper, map_lower)
                holder.append((index, cur_lower, max_overlap, None))
                cur_lower = max_overlap
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
            elif cur_lower >= map_lower:
                logger.debug(f"FLAG 2")
                # Calculate the furthest overlap
                furthest_overlap = min(cur_upper, map_upper)
                tag = self.mapping[self.keys[p_mapping]]
                holder.append((index, cur_lower, furthest_overlap, tag))
                map_lower = furthest_overlap
                cur_lower = furthest_overlap
                logger.debug(f"Overlap found at {furthest_overlap}, tag={tag}")
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
            "activity_rate"  # (Rates do not need to be distributed)
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
        for index, start, end, tag in breakdown:
            row = dataset.loc[index]
            segment_data = {}
            segment_duration = (end - start).total_seconds()

            # How much of the whole dataset is the segment
            segment_prop = segment_duration / row["Duration (s)"]
            try:
                assert(segment_prop <= 1)
            except AssertionError:
                print(f"Segment prop of value={segment_prop} > 1")
                print(f"index={index}, start={start}, end={end}, tag={tag}")
                print(f"row={row.to_dict()}")
            # Determine which rows to keep/normalize
            for key, item in row.to_dict().items():
                if key in blacklist:
                    continue
                if key in non_numeric:
                    segment_data[key] = item
                # if numeric, we need to normalize
                else:
                    logger.debug(f"Normalizing {key}")
                    try:
                        segment_data[key] = item * segment_prop
                    except:
                        logger.error(f"Unable to proportionize key={key} with value={item}")
                        raise

            # Now we add in all the useful data
            segment_data["start"] = start
            segment_data["end"] = end
            segment_data["tag"] = tag
            segment_data["duration"] = (end - start).total_seconds()
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
        dataset = self.supplement_dataset(dataset)
        breakdown = self.breakdown_dataset(dataset)
        compiled = self.compile_data(breakdown, dataset)
        return compiled


if __name__ == "__main__":
    from datetime import datetime
    import sys
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
        {"datetime": datetime(2000, 1, 1, 1, 1, 0), "Duration (s)": 10},
        # Fully contained
        {"datetime": datetime(2000, 1, 1, 1, 1, 10), "Duration (s)": 10},
        # Runoff on the right crosses two tag fields
        {"datetime": datetime(2000, 1, 1, 1, 1, 20), "Duration (s)": 15},
        # Runoff on the right into the void
        {"datetime": datetime(2000, 1, 1, 1, 1, 35), "Duration (s)": 10}
    ])
    dt_test = PointerDataTagger(ann)
    dt_breakdown = dt_test.breakdown_dataset(test_dataset)
    # Unit tests for the breakdown
    assert(len(dt_breakdown) == 7)
    assert([i[-1] for i in dt_breakdown] == [None, 1, 2, 3, 4, 4, None])
    # TEST THE DURATION (NOT IMPLEMENTED)
    # assert
    # Tests datasets completely outside the annotations
    ann = pd.DataFrame([
        {"start": datetime(2000, 1, 1, 1, 1, 15), "end": datetime(2000, 1, 1, 1, 1, 20), "tag_details": 1},
        {"start": datetime(2000, 1, 1, 1, 1, 20), "end": datetime(2000, 1, 1, 1, 1, 30), "tag_details": 2},
    ])
    test_dataset = pd.DataFrame([
        # This is completely removed
        {"datetime": datetime(2000, 1, 1, 1, 1, 0), "Duration (s)": 10},
        # This is missing stuff on the left but the right fits in neatly
        {"datetime": datetime(2000, 1, 1, 1, 1, 10), "Duration (s)": 10},
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
