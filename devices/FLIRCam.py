import sys
sys.path.append('../../')
import PySpin
from PySpin import CameraPtr


class FLIRCam:
    def __init__(self, exp_time, avi_type, saving_path):
        self.exp_time = exp_time
        self.avi_type = avi_type
        self.saving_path = saving_path

        self.cam, self.nodemap, self.cam_list, self.system = self.find_cam()
        self.avi_recorder = self.create_avi_recorder(self.saving_path)

    def find_cam(self):
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()
        if num_cameras == 0:
            cam_list.Clear()
            # Release system instance
            system.ReleaseInstance()
            print("Couldn't detect any FLIR cameras")

        else:
            cam = cam_list[0]
            cam.Init()
            nodemap = cam.GetNodeMap()
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

                # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            return cam, nodemap, cam_list, system

    def create_avi_recorder(self, path):

        node_acquisition_framerate = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(node_acquisition_framerate) and not PySpin.IsReadable(node_acquisition_framerate):
            print('Unable to retrieve frame rate. Aborting...')
            return False
        framerate_to_set = node_acquisition_framerate.GetValue()
        if self.avi_type == "UNCOMPRESSED":
            option = PySpin.AVIOption()
            option.frameRate = framerate_to_set

        elif self.avi_type == "MJPG":
            option = PySpin.MJPGOption()
            option.frameRate = framerate_to_set
            option.quality = 75

        elif self.avi_type == "H264":
            option = PySpin.H264Option()
            option.frameRate = framerate_to_set
            option.bitrate = 1000000
            # option.height = images[0].GetHeight()
            # option.width = images[0].GetWidth()

        avi_recorder = PySpin.SpinVideo()
        avi_recorder.Open(path, option)
        return avi_recorder

    def start_acquisition(self):
        self.cam.BeginAcquisition()

    def stop_acquisition(self):
        self.cam.EndAcquisition()

    def grab_frame(self):
        frame = self.cam.GetNextImage(self.exp_time)
        return frame.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

    def save_to_avi(self, frame):
        self.avi_recorder.Append(frame)

    def close(self):
        self.avi_recorder.Close()

        self.cam.DeInit()
        self.cam = None
        self.cam_list.Clear()
        self.system.ReleaseInstance()

