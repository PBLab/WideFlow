from analysis.utils.extract_from_metadata_file import extract_from_metadata_file
import numpy as np
import matplotlib.pyplot as plt


base_path = '/data/Lena/WideFlow_prj'

mice = ['21ML','31MN','54MRL','63MR','64ML']

dates_vec = ['20230604','20230611', '20230612', '20230613', '20230614', '20230615']
#dates_vec = ['20230604','20230611', '20230612', '20230613', '20230614']
#dates_vec = ['20230618', '20230619', '20230620', '20230621', '20230622']

#sessions_vec = ['spont_mockNF_excluded_closest','NF1', 'NF2', 'NF3', 'NF4']
sessions_vec = ['spont_mockNF_excluded_closest','NF1', 'NF2', 'NF3', 'NF4', 'NF5']
#sessions_vec = ['spont_mockNF_ROI2_excluded_closest','NF21', 'NF22', 'NF23', 'NF24', 'NF25']

NF_sess_frames = 65000
spont_sess_frames = 50000
set_threshold = (np.arange(0.5, 4.5, 0.1)).tolist()

peaks = np.zeros((len(mice),len(sessions_vec)-1))

for mouse_id in mice:
    crossings = np.zeros((len(sessions_vec), len(set_threshold)))
    for date, session_id in zip(dates_vec, sessions_vec):
        #timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{mouse_id}/{session_id}/metadata.txt')
        timestamp, cue, metric_result, threshold, serial_readout = extract_from_metadata_file(f'{base_path}/{date}/{mouse_id}/{date}_{mouse_id}_{session_id}/metadata.txt')
        #count = np.zeros(len(set_threshold))
        #rewards = np.sum(cue)
        for i in set_threshold:
            for value in metric_result:
                if value > i:
                    crossings[sessions_vec.index(session_id), set_threshold.index(i)] += 1


    first_row = crossings[0,:]
    first_row = (NF_sess_frames/spont_sess_frames)*first_row
    crossings[0,:] = first_row
    norm_crossings = crossings - first_row[np.newaxis,:]
    norm_crossings = [[element / first_row[i] for element in column] for i, column in enumerate(norm_crossings)]
    #crossings[0,:] = (NF_sess_frames/spont_sess_frames)*crossings[0,:]

    for i in range(len(norm_crossings)-1):
        #max_abs = np.max(np.abs(norm_crossings[i+1]))
        max_abs = np.max(norm_crossings[i + 1])
        #abslt_crossings = list(np.abs(norm_crossings[i+1]))
        abslt_crossings = list(norm_crossings[i + 1])
        thresh_inx = abslt_crossings.index(max_abs)
        peaks[mice.index(mouse_id),i] = set_threshold[thresh_inx]
        a=5




peaks_flat = [item for sublist in list(peaks) for item in sublist]

hist, bin_edges = np.histogram(peaks_flat, bins=7)
mode_bin_index = np.argmax(hist)
mode_values = [(bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2]
mean_peaks = np.mean(peaks_flat)

plt.hist(peaks_flat, bins=7, edgecolor = 'black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Mode Highlighted')
plt.axvline(x=mode_values[0], color='r', linestyle='--', label=f'Mode = {mode_values[0]:.2f}')
plt.axvline(x=mean_peaks,color = 'darkred', linestyle='--', label =f'Mean = {mean_peaks:.2f}' )
plt.legend()
#plt.show()
# color_palette = plt.cm.get_cmap('plasma', len(sessions_vec)) #colormap options https://matplotlib.org/stable/tutorials/colors/colormaps.html
# for i, session_id in zip(norm_crossings, sessions_vec):
#     plt.plot(set_threshold,i, color =color_palette(sessions_vec.index(session_id)), label = f'{session_id}')
#
#
# plt.grid()
# plt.legend()
# plt.title(f'{mouse_id}_{roi}')
# plt.xlabel ('Threshold')
# plt.ylabel ('Threshold crossing rate (a.u.)')
# plt.show()
#plt.savefig(f'{base_path}/Figures_exp2_all_mice_compare/{mouse_id}_{roi}_rate_vs_thrsh_all_sessions.png',dpi=500)
plt.savefig(f'{base_path}/Figs_for_paper/TCR_peaks_hist.svg',format='svg',dpi=500)




#[item[0] for item in norm_crossings]
a=5
#print(count, rewards)
# b = norm_crossings[1]
# b_abslt = list(np.abs(b))
# index = b_abslt.index(a[1])
# thresh = set_threshold[index]