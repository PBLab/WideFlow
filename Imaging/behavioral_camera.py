


def behavioral_camera(cam, query):
    cam.start_acquisition
    while True:
        if query.empty():
            continue
        q = query.get()
        if q == "grab":
            frame = cam.grab_frame()
            cam.save_to_avi(frame)
        elif q == "terminate":
            print("live video terminating")
            terminate_process()
            break
        else:
            raise KeyError(f'behavioral_camera query "{q}" is invalid')

        def terminate_process():
            cam.stop_acquisition()
            cam.close()