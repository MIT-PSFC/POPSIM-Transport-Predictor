# Shots that feature our Ip ramp that we can use to characterize ranges of inputs
IP_RAMP_SHOTS = {
    199121: {"start": 3.0, "end": 5.0},
    199122: {"start": 2.5, "end": 5.0},
    199125: {"start": 2.0, "end": 5.5},
    199126: {"start": 2.0, "end": 4.3},
    # These 2019XX series are the ones I'll be mostly targeting, still using the above to see what's possible though
    201907: {"start": 1.5, "end": 5.2},
    201908: {"start": 1.5, "end": 5.3},
    201910: {"start": 1.5, "end": 5.4},
    201911: {"start": 1.5, "end": 5.4},
    201912: {"start": 1.5, "end": 5.4},
    201913: {"start": 1.5, "end": 5.4},
    201914: {"start": 1.5, "end": 5.4},
    201927: {"start": 1.5, "end": 4.7},  # OUR BASE SHOT
    201934: {"start": 1.5, "end": 5.4},
}

# Shots that have our exact feedback control scheme (matching 201927)
# that we can use to characterize typical error between our target signal and the actual signal
FEEDBACK_CONTROL_SHOTS = {201927: {"start": 0.7, "end": 4.7}}
