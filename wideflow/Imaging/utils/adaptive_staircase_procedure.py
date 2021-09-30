def fixed_step_staircase_procedure(threshold, cues_seq, activation_cue, typical_n, typical_count, step):
    """

    Args:
        threshold: float - current threshold
        cues_seq: list - a list of zeros and ones of length "frame_counter"
        cue: int - constant, num indicating activity
        typical_n: int - typical number of frames were a reward should be given
        typical_count: int - typical number of times metric is above threshold in typical_n frames
        step: float - threshold update delta

    Returns: threshold: float - updated threshold

    """
    n = cues_seq[::-1].index(activation_cue)
    if n > typical_n:
        return threshold - step
    elif cues_seq[-typical_n:].count(activation_cue) > typical_count:
        return threshold + step
    else:
        return threshold



