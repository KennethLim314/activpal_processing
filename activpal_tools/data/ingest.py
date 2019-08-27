"""Reads data into a cross-comparable dataset ready for comparison
takes in activepal data and spits out a coherent, consistent dataset
Note that we might want to analyze on a few levels

Data may come in in day level and epoch level (ns)

For day-level datasets, they should breakdown into the following columns
    user_id:
    mvpa
    lipa
    sb
    sleep
    aktivo_score
    datetime
"""

import numpy as np
import logging
from activpal_tools.data.processing import PointerDataTagger

logger = logging.getLogger(__name__)


def ingest_phone():
    pass


def build_gold_standard(activpal_data, sleep_annotations,
                        mvpa_step_cutoff=100):
    """Builds the gold standard dataset

    Args:
        activpal_data (DataFrame): Description

        sleep_annotations (TYPE): Description
    """
    # Acquire key columns of activpal data
    activpal_data = activpal_data.copy().loc[:, [
        "start",
        "end",
        "duration",
        "steps",
        "point_cadence",
        "activpal_event"
    ]]
    activpal_data["tag"] = "sb"
    # Create LIPA, MVPA
    # LIPA is in the case where any stepping is happening
    sel = activpal_data["activpal_event"] == "stepping"
    activpal_data.loc[sel, "tag"] = "lipa"
    # MVPA then overrides that
    sel = np.logical_and(
        activpal_data["activpal_event"] == "stepping",
        activpal_data["point_cadence"] > mvpa_step_cutoff
    )
    activpal_data.loc[sel, "tag"] = "mvpa"

    # Create initial sleep tag annotations with the PDT
    pdt = PointerDataTagger(sleep_annotations)
    res = pdt.tag_data(activpal_data)

    return res


if __name__ == "__main__":
    import pandas as pd
    import os
    from datetime import datetime

    from activpal_tools.misc import test_data_dir
    from activpal_tools.data.transform import to_epochs


    parse_dates = ["start", "end"]
    test_apd = pd.read_csv(os.path.join(test_data_dir, "test_gold-std-creation_activpal-data.csv"),
                           parse_dates=parse_dates)
    test_sleep_ann = pd.read_csv(os.path.join(test_data_dir, "test_gold-std-creation_sleep-ann.csv"),
                                 parse_dates=parse_dates)
    print(test_apd.head())
    gs = build_gold_standard(test_apd, test_sleep_ann)
    print(gs.loc[:, ["start", "end", "tag"]])

    gs_epochs = to_epochs(gs, resolution=15)
    print(gs_epochs)
    # ADD AUTOMATED TETSTS
