def extract_from_metadata_file(path):
    timestamp = []
    cue = []
    metric_result = []
    serial_readout = []
    with open(path, "r") as f:
        line_txt = ""
        while line_txt != "frames metadata:\n":
            line_txt = f.readline()
        for line_txt in f:
            line_txt = line_txt.split('    ')
            timestamp.append(float(line_txt[0].split(':')[1]))
            cue.append(int(line_txt[1].split(':')[1]))
            metric_result.append(float(line_txt[2].split(':')[1]))
            serial_readout.append(int(line_txt[3].split(':')[1]))

    return timestamp, cue, metric_result, serial_readout
