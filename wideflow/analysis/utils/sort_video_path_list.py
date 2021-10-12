def sort_video_path_list(vlist):
    sort_vlist = [None] * len(vlist)
    for path in vlist:
        idx = int(path.split('_')[-1].split('.')[0])
        sort_vlist[idx] = path

    return sort_vlist
