
def deinterleave(video, n_channels):
    deinterleav_data = {}
    for i in range(n_channels):
        deinterleav_data[f'channel_{i}'] = video[i::n_channels, :, :]

    return deinterleav_data