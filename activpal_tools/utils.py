from datetime import datetime, timedelta
from activpal_tools import misc
from collections import defaultdict
from numpy import isclose
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def dt2float(dt):
    res = (dt - datetime(1899, 12, 30)).total_seconds() / (24 * 60 * 60)
    return res
