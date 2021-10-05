from devices.FLIRCam import FLIRCam
import PySpin


def run_triggered_behavioral_camera(query, saving_path, **camera_config):
    bcam = FLIRCam(**camera_config)
    bcam.avi_recorder.Open(saving_path, bcam.avi_recorder_options)
    bcam.start_acquisition()
    cap = False
    while cap is False:
        print('accuire bcam first frame')
        cap, prev_frame = bcam.grab_frame()

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
                    prev_frame = frame
                else:
                    # query.put('grab')
                    bcam.save_to_avi(prev_frame)
            elif q == 'finish':
                break

    print('ending behavioral camera acquisition')
    bcam.stop_acquisition()
    bcam.close()