import numpy as np


def evaluate_session_learning(cues, response, delta_t):
    cues = np.array(cues)
    response = np.array(response)

    n_t = len(response)
    n_response = np.sum(response)
    p_response = n_response / n_t

    cues_inds = np.where(cues == 1)[0]
    n_cues = len(cues_inds)
    cue_response, cue_no_response, unknown_response = 0, 0, 0
    for idx in cues_inds:
        if 1 not in response[idx-delta_t: idx + 1] and 1 in response[idx + 1: idx + delta_t]:
            cue_response += 1
        if 1 not in response[idx + 1: idx + delta_t]:
            cue_no_response += 1
        if 1 in response[idx - delta_t: idx + 1] and 1 in response[idx + 1: idx + delta_t]:
            unknown_response += 1

    p_cue_response = cue_response / (n_cues - unknown_response)

    return p_cue_response / p_response


def evaluate_intra_session_learning(cues, response, delta_t, eval_n_cues):
    cues_inds = np.where(cues == 1)[0]
    session_eval = [None] * len(cues_inds[eval_n_cues:])
    prev_idx = 0
    for i in range(eval_n_cues, len(cues_inds)):
        idx = cues_inds[i]
        session_eval[i] = evaluate_session_learning(cues[prev_idx: idx + delta_t], response[prev_idx: idx + delta_t], delta_t)
        prev_idx = cues_inds[i - eval_n_cues]

    return session_eval




