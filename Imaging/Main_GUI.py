import wx
from Imaging.VideoCapture import VideoCapture
from Imaging.gui_frames import AcquisitionConfig
from Imaging.main import run_session
from pubsub.pub import Publisher

from pyvcam import pvc
from devices.PVCam import PVCamera


class Main_GUI(wx.Frame):
    def __init__(self, cam, *args, **kwargs):
        super(Main_GUI, self).__init__(*args, **kwargs)
        self.cam = cam
        self.configuration = None
        self.InitUI()

    def InitUI(self):
        self.menu_bar()
        self.main_layout()

        self.Move((0, 0))
        self.SetSize((300, 300))
        self.SetTitle('WideFlow')
        self.Centre()

    def menu_bar(self):
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        file_quit_Item = fileMenu.Append(wx.ID_EXIT, 'Quit', 'Quit application')
        file_save_Item = fileMenu.Append(wx.ID_SAVE, 'Save', 'Save Session')
        file_load_Item = fileMenu.Append(wx.ID_OPEN, 'Configuration', 'Load Session Configurations')

        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)

        self.Bind(wx.EVT_MENU, self.OnQuit, file_quit_Item)
        self.Bind(wx.EVT_MENU, self.OnSave, file_save_Item)
        self.Bind(wx.EVT_MENU, self.OnConfiguration, file_load_Item)

    def OnQuit(self, e):
        self.Close()

    def OnSave(self):
        pass

    def OnConfiguration(self):
        pass

    def main_layout(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour("gray")

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(9)

        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        st1 = wx.StaticText(panel, label='RAW')
        st1.SetFont(font)
        hbox1.Add(st1, flag=wx.RIGHT, border=8)
        vbox.Add((-1, 10))

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        btn_start = wx.Button(panel, label='Start Session', size=(70, 30))
        hbox2.Add(btn_start)
        btn_stop = wx.Button(panel, label='Stop', size=(70, 30))
        hbox2.Add(btn_stop, flag=wx.LEFT | wx.BOTTOM, border=5)
        vbox.Add(hbox2, flag=wx.ALIGN_RIGHT | wx.RIGHT, border=10)
        vbox.Add((-1, 25))

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        btn_show = wx.Button(panel, label='Show\nLive', size=(70, 50))
        hbox3.Add(btn_show)
        vbox.Add(hbox3, flag=wx.ALIGN_RIGHT | wx.RIGHT, border=10)
        vbox.Add((-1, 25))

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        btn_config = wx.Button(panel, label='Load\nConfigurations', size=(85, 50))
        hbox4.Add(btn_config)
        vbox.Add(hbox4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
        vbox.Add((-1, 25))

        panel.SetSizer(vbox)

        self.Bind(wx.EVT_BUTTON, self.open_video_window, btn_show)
        self.Bind(wx.EVT_BUTTON, self.open_acquisition_configuration_window, btn_config)
        self.Bind(wx.EVT_BUTTON, self.start_acquisition, btn_start)

    def open_video_window(self, event):
        video_window = VideoCapture(parent=None, title='Video', cam=self.cam)
        video_window.Show()

    def open_acquisition_configuration_window(self, event):
        acquisition_config_window = AcquisitionConfig(parent=None, title='Session Acquisition Configurations',
                                                      config=self.configuration)
        acquisition_config_window.Show()

    def start_acquisition(self, event):
        try:
            run_session(self.config, self.cam)
        except:
            print("An exception occurred")
            self.cam.close()

import numpy as np
class Camera:
    def __init__(self):
        self.sensor_size = (512, 512)
        self.exp_time = 10

    def open(self):
        pass

    def close(self):
        pass

    def get_frame(self):
        return np.random.random(self.sensor_size)*255

    def start_live(self):
        pass

def main():
    cam = Camera()
    # pvc.init_pvcam()
    # cam = next(PVCamera.detect_camera())
    app = wx.App()
    main_gui = Main_GUI(parent=None, title='blabla', cam=cam)
    main_gui.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()

