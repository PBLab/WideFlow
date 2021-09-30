from devices.FLIRCam import FLIRCam


def run_triggered_behavioral_camera(query, saving_path, **camera_config):
    cam = FLIRCam(**camera_config)
    cam.avi_recorder.Open(saving_path, cam.avi_recorder_options)
    cam.start_acquisition()
    print('starting behavioral camera acquisition')
    while True:
        if not query.empty():
            q = query.get()
            if q == 'grab':
                cap, frame = cam.grab_frame()
                if cap:
                    cam.save_to_avi(frame)
                    prev_frame = frame
                else:
                    cam.save_to_avi(prev_frame)
            elif q == 'finish':
                break

    print('ending behavioral camera acquisition')
    cam.stop_acquisition()
    cam.close()