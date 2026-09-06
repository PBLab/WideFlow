# path may also be an already-open text stream, not just a file path -- added by Claude, 2026-08-12
def extract_from_metadata_file(path):
    timestamp = []
    cue = []
    metric_result = []
    threshold = []
    serial_readout = []
    trial_number = []
    sr_prv = 1

    if isinstance(path, str):
        f = open(path, "r")
        close_when_done = True
    else:
        f = path
        close_when_done = False

    try:
        line_txt = ""
        while line_txt != "frames metadata:\n":
            line_txt = f.readline()
        for line_txt in f:
            line_txt = line_txt.split('    ')
            timestamp.append(float(line_txt[0].split(':')[1]))
            cue.append(int(line_txt[1].split(':')[1]))
            metric_result.append(float(line_txt[2].split(':')[1]))
            threshold.append(float(line_txt[3].split(':')[1]))
            if len(line_txt)==6 and line_txt[5].split(':')[0] == 'trial number':
                trial_number.append(float(line_txt[5].split(':')[1]))

            sr = line_txt[4].split(':')[1][0]
            if sr.isnumeric():  # handle readout failure - "?"
                sr_prv = int(sr)
                serial_readout.append(sr_prv)
            else:
                serial_readout.append(sr_prv)
    finally:
        if close_when_done:
            f.close()

    return timestamp, cue, metric_result, threshold, serial_readout, trial_number
