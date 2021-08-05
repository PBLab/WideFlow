from copy import copy


def load_matching_points(path):
    with open(path) as f:
        match_p_src_text = f.readline()
        match_p_dst_text = f.readline()

    match_p_src_text = match_p_src_text.split(': ')[1]
    match_p_src_text = match_p_src_text.replace("[", ",")
    match_p_src_text = match_p_src_text.replace("]", ",")
    match_p_src_text = match_p_src_text.split(',')

    match_p_src = []
    yx = [None, None]
    ind = -1
    for word in match_p_src_text:
        try:
            float(word)
            float_flag = True
            ind += 1
        except:
            float_flag = False

        if float_flag:
            yx[ind % 2] = float(word)
            if ind % 2:
                match_p_src.append(copy(yx))

    match_p_dst_text = match_p_dst_text.split(': ')[1]
    match_p_dst_text = match_p_dst_text.replace("[", ",")
    match_p_dst_text = match_p_dst_text.replace("]", ",")
    match_p_dst_text = match_p_dst_text.split(',')

    match_p_dst = []
    yx = [None, None]
    ind = -1
    for word in match_p_dst_text:
        try:
            float(word)
            float_flag = True
            ind += 1
        except:
            float_flag = False

        if float_flag:
            yx[ind % 2] = float(word)
            if ind % 2:
                match_p_dst.append(copy(yx))

    return match_p_src, match_p_dst