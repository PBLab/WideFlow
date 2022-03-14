from devices.FLIRCam import FLIRCam
import PySpin

import numpy as np
from multiprocessing import shared_memory


def run_triggered_behavioral_camera(query, saving_path, shm_name, **camera_config):
    bcam = FLIRCam(**camera_config)
    bcam.avi_recorder.Open(saving_path, bcam.avi_recorder_options)
    bcam.start_acquisition()
    cap = False
    while cap is False:
        print('accuire bcam first frame')
        cap, frame = bcam.grab_frame()

    if shm_name is not None:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        frame_shm = np.ndarray(shape=frame.shape,
                           dtype=frame.dtype, buffer=existing_shm.buf)
    else:
        frame_shm = frame

    while True:
        if not query.empty():
            q = query.get()
            if q == "start":
                break

    print('starting behavioral camera acquisition')
    while True:
        if bcam.chosen_trigger == "SOFTWARE":
            node_softwaretrigger_cmd = PySpin.CCommandPtr(bcam.nodemap.GetNode('TriggerSoftware'))
            if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
                print('Unable to execute trigger. Aborting...')
                return False
            node_softwaretrigger_cmd.Execute()

        if not query.empty():
            q = query.get()
            if q == 'grab':
                cap, frame = bcam.grab_frame()
                if cap:
                    bcam.save_to_avi(frame)
                    frame_shm[:] = frame
                else:
                    # query.put('grab')
                    bcam.save_to_avi(frame_shm[:])
            elif q == 'finish':
                break

    print('ending behavioral camera acquisition')
    bcam.stop_acquisition()
    bcam.close()