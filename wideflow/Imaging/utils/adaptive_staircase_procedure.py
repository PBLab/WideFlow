def fixed_step_staircase_procedure(threshold, cues_seq, cue, typical_n, typical_count, step):
    n = cues_seq[::-1].index(cue)
    if n > typical_n:
        return threshold - step
    elif cues_seq[-typical_n:].count(cue) > typical_count:
        return threshold + step
    else:
        return threshold



