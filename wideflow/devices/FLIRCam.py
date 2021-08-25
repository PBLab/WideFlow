import sys
sys.path.append('../../../')
import PySpin
from PySpin import CameraPtr


class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2



class FLIRCam:
    def __init__(self, exp_time, roi_bbox=None, avi_type="MJPG", acquisition_mode=None, chosen_trigger='HARDWARE'):
        self.exp_time = exp_time
        self.roi_bbox = roi_bbox
        self.avi_type = avi_type
        self.acquisition_mode = acquisition_mode
        self.chosen_trigger = chosen_trigger
        self.CHOSEN_TRIGGER = getattr(TriggerType, self.chosen_trigger)

        self.find_cam()

        if self.roi_bbox != None:
            if self.configure_image_roi() is False:
                print("couldn't set image roi")

        if self.acquisition_mode == 'TRIGGER':
            if self.configure_trigger() is False:
                print("couldn't initialize trigger mode")

        self.avi_recorder, self.avi_recorder_options = self.create_avi_recorder()

    def find_cam(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()
        if num_cameras == 0:
            self.cam_list.Clear()
            # Release system instance
            self.system.ReleaseInstance()
            print("Couldn't detect any FLIR cameras")

        else:
            self.cam = self.cam_list[0]
            self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
            self.cam.Init()
            self.nodemap = self.cam.GetNodeMap()

    def configure_image_roi(self):
        try:
            node_offset_x = PySpin.CIntegerPtr(self.nodemap.GetNode('OffsetX'))
            if PySpin.IsAvailable(node_offset_x) and PySpin.IsWritable(node_offset_x):
                node_offset_x.SetValue(self.roi_bbox[1])
                print('Offset X set to %i...' % node_offset_x.GetMin())
            else:
                print('Offset X not available...')

            node_offset_y = PySpin.CIntegerPtr(self.nodemap.GetNode('OffsetY'))
            if PySpin.IsAvailable(node_offset_y) and PySpin.IsWritable(node_offset_y):
                node_offset_y.SetValue(self.roi_bbox[0])
                print('Offset Y set to %i...' % node_offset_y.GetMin())
            else:
                print('Offset Y not available...')

            node_width = PySpin.CIntegerPtr(self.nodemap.GetNode('Width'))
            if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
                width_to_set = self.roi_bbox[2]
                node_width.SetValue(width_to_set)
                print('Width set to %i...' % node_width.GetValue())
            else:
                print('Width not available...')

            node_height = PySpin.CIntegerPtr(self.nodemap.GetNode('Height'))
            if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
                height_to_set = self.roi_bbox[3]
                node_height.SetValue(height_to_set)
                print('Height set to %i...' % node_height.GetValue())
            else:
                print('Height not available...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    def configure_trigger(self):
        """
        This function configures the camera to use a trigger. First, trigger mode is
        set to off in order to select the trigger source. Once the trigger source
        has been selected, trigger mode is then enabled, which has the camera
        capture only a single image upon the execution of the chosen trigger.

         :param cam: Camera to configure trigger for.
         :type cam: CameraPtr
         :return: True if successful, False otherwise.
         :rtype: bool
        """

        print('*** CONFIGURING TRIGGER ***\n')

        print(
            'Note that if the application / user software triggers faster than frame time, the trigger may be dropped / skipped by the camera.\n')
        print(
            'If several frames are needed per trigger, a more reliable alternative for such case, is to use the multi-frame mode.\n\n')

        if self.CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            print('Software trigger chosen ...')
        elif self.CHOSEN_TRIGGER == TriggerType.HARDWARE:
            print('Hardware trigger chose ...')

        try:
            # Ensure trigger mode off
            # The trigger must be disabled in order to configure whether the source
            # is software or hardware.
            nodemap = self.cam.GetNodeMap()
            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
            if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
                print('Unable to disable trigger mode (node retrieval). Aborting...')
                return False

            node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
            if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
                print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

            print('Trigger mode disabled...')

            # Set TriggerSelector to FrameStart
            # For this example, the trigger selector should be set to frame start.
            # This is the default for most cameras.
            node_trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
            if not PySpin.IsAvailable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
                print('Unable to get trigger selector (node retrieval). Aborting...')
                return False

            node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
            if not PySpin.IsAvailable(node_trigger_selector_framestart) or not PySpin.IsReadable(
                    node_trigger_selector_framestart):
                print('Unable to set trigger selector (enum entry retrieval). Aborting...')
                return False
            node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())

            print('Trigger selector set to frame start...')

            # Select trigger source
            # The trigger source must be set to hardware or software while trigger
            # mode is off.
            node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
            if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
                print('Unable to get trigger source (node retrieval). Aborting...')
                return False

            if self.CHOSEN_TRIGGER == TriggerType.SOFTWARE:
                node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
                if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                        node_trigger_source_software):
                    print('Unable to set trigger source (enum entry retrieval). Aborting...')
                    return False
                node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
                print('Trigger source set to software...')

            elif self.CHOSEN_TRIGGER == TriggerType.HARDWARE:
                node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
                if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                        node_trigger_source_hardware):
                    print('Unable to set trigger source (enum entry retrieval). Aborting...')
                    return False
                node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
                print('Trigger source set to hardware...')

            # Turn trigger mode on
            # Once the appropriate trigger source has been set, turn trigger mode
            # on in order to retrieve images using the trigger.
            node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
            if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
                print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
            print('Trigger mode turned back on...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    def reset_trigger(self):
        """
        This function returns the camera to a normal state by turning off trigger mode.

        :param nodemap: Transport layer device nodemap.
        :type nodemap: INodeMap
        :returns: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            node_trigger_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('TriggerMode'))
            if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
                print('Unable to disable trigger mode (node retrieval). Aborting...')
                return False

            node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
            if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
                print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

            print('Trigger mode disabled...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    def create_avi_recorder(self):

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

        avi_recorder = PySpin.SpinVideo()

        return avi_recorder, option

    def start_acquisition(self):
        node_acquisition_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
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

        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        self.cam.BeginAcquisition()

    def stop_acquisition(self):
        self.cam.EndAcquisition()

    def grab_frame(self):
        try:
            image_result = self.cam.GetNextImage(self.exp_time)
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            else:
                frame = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                image_result.Release()
                return True, frame
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False, None

    def save_to_avi(self, frame):
        self.avi_recorder.Append(frame)

    def close(self):
        self.avi_recorder.Close()

        if self.acquisition_mode == 'TRIGGER':
            self.reset_trigger()

        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()


