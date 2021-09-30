def fixed_step_staircase_procedure(threshold, cues_seq, cue, typical_n, typical_count, step):
    try:
        n = cues_seq[::-1].index(cue)
    except:
        n = len(cues_seq)

    if n > typical_n:
        return threshold - step
    elif cues_seq[-typical_n:].count(cue) > typical_count:
        return threshold + step
    else:
        return threshold



