def extract_from_metadata_file(path):
    timestamp = []
    cue = []
    metric_result = []
    serial_readout = []
    sr_prv = 1
    with open(path, "r") as f:
        line_txt = ""
        while line_txt != "frames metadata:\n":
            line_txt = f.readline()
        for line_txt in f:
            line_txt = line_txt.split('    ')
            timestamp.append(float(line_txt[0].split(':')[1]))
            cue.append(int(line_txt[1].split(':')[1]))
            metric_result.append(float(line_txt[2].split(':')[1]))
            sr = line_txt[3].split(':')[1][0]
            if sr.isnumeric():  # handle readout failure - "?"
                sr_prv = int(sr)
                serial_readout.append(sr_prv)
            else:
                serial_readout.append(sr_prv)

    return timestamp, cue, metric_result, serial_readout
