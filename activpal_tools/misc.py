import os
# constants

# things for the pointerdataTagger
non_numeric = {
    "Data Count",
    "activity_rate",  # (Rates do not need to be distributed)
    "Event Type",  # Encoded as integer by activepal
    "activpal_event",
    "trial",
    "trial_set",
    "tag"
}

blacklist = {
    "Time",
    "Time(approx)",
    "datetime",
    "Duration (s)",
    "Cumulative Step Count",
    "Waking Day",
    "start",
    "end",
}


test_data_dir = os.path.join(
    os.path.dirname(__file__),
    "test_data")
# print(test_data_dir)
