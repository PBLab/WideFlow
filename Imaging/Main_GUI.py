import wx
from Imaging.VideoCapture import VideoCapture, ProcessingWindow
from Imaging.gui_frames import AcquisitionConfig, CameraConfig
from pubsub import pub

from devices.PVCam import PVCamera
from pyvcam import pvc

from core.processing import *
from core.metric import *


class Main_GUI(wx.Frame):
    def __init__(self, cam, *args, **kwargs):
        super(Main_GUI, self).__init__(*args, **kwargs)
        self.cam = cam
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

        #  raw video vertical box
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        st1 = wx.StaticText(panel, label='RAW')
        st1.SetFont(font)
        hbox1.Add(st1, flag=wx.RIGHT, border=8)
        vbox.Add((-1, 10))

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        btn_start = wx.Button(panel, label='Start', size=(70, 30))
        hbox2.Add(btn_start)
        btn_pause = wx.Button(panel, label='Pause', size=(70, 30))
        hbox2.Add(btn_pause, flag=wx.LEFT | wx.BOTTOM, border=5)
        btn_stop = wx.Button(panel, label='Stop', size=(70, 30))
        hbox2.Add(btn_stop, flag=wx.LEFT | wx.BOTTOM, border=5)
        vbox.Add(hbox2, flag=wx.ALIGN_RIGHT | wx.RIGHT, border=10)
        vbox.Add((-1, 25))

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        btn_show = wx.Button(panel, label='show', size=(70, 30))
        hbox3.Add(btn_show)
        vbox.Add(hbox3, flag=wx.ALIGN_RIGHT | wx.RIGHT, border=10)
        vbox.Add((-1, 25))

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        st1 = wx.StaticText(panel, label='frame rate')
        st1.SetFont(font)
        hbox4.Add(st1, flag=wx.RIGHT, border=8)
        tc = wx.TextCtrl(panel)
        hbox4.Add(tc, proportion=1)
        vbox.Add(hbox4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
        vbox.Add((-1, 25))

        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        cb1 = wx.CheckBox(panel, label='option 1')
        cb1.SetFont(font)
        hbox5.Add(cb1)
        cb2 = wx.CheckBox(panel, label='option 2')
        cb2.SetFont(font)
        hbox5.Add(cb2, flag=wx.LEFT, border=10)
        cb3 = wx.CheckBox(panel, label='option 3')
        cb3.SetFont(font)
        hbox5.Add(cb3, flag=wx.LEFT, border=10)
        vbox.Add(hbox5, flag=wx.LEFT, border=10)
        vbox.Add((-1, 25))

        panel.SetSizer(vbox)

        self.Bind(wx.EVT_BUTTON, self.open_video_window(), btn_show)
        self.Bind(wx.EVT_BUTTON, self.start_acquisition(), btn_start)
        self.Bind(wx.EVT_BUTTON, self.stop_acquisition(), btn_stop)


    def open_video_window(self):
        video_window = VideoCapture(parent=None, title='Video', cam=self.cam)
        video_window.Show()

    def open_acquisition_config_window(self):
        acquisition_config_window = AcquisitionConfig(parent=None, title='Session Acquisition Configurations')
        acquisition_config_window.Show()

    def open_camera_config_window(self):
        camera_config_window = CameraConfig(parent=None, title='Camera Configurations', cam=self.cam)
        camera_config_window.Show()

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass


# class Example(wx.Frame):
#     def __init__(self, parent, title):
#         super(Example, self).__init__(parent, title=title)
#
#         self.InitUI()
#         self.Centre()
#
#     def InitUI(self):
#
#         panel = wx.Panel(self)
#
#         font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
#
#         font.SetPointSize(9)
#
#         vbox = wx.BoxSizer(wx.VERTICAL)
#
#         hbox1 = wx.BoxSizer(wx.HORIZONTAL)
#         st1 = wx.StaticText(panel, label='Class Name')
#         st1.SetFont(font)
#         hbox1.Add(st1, flag=wx.RIGHT, border=8)
#         tc = wx.TextCtrl(panel)
#         hbox1.Add(tc, proportion=1)
#         vbox.Add(hbox1, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)
#
#         vbox.Add((-1, 10))
#
#         hbox2 = wx.BoxSizer(wx.HORIZONTAL)
#         st2 = wx.StaticText(panel, label='Matching Classes')
#         st2.SetFont(font)
#         hbox2.Add(st2)
#         vbox.Add(hbox2, flag=wx.LEFT | wx.TOP, border=10)
#
#         vbox.Add((-1, 10))
#
#         hbox3 = wx.BoxSizer(wx.HORIZONTAL)
#         tc2 = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
#         hbox3.Add(tc2, proportion=1, flag=wx.EXPAND)
#         vbox.Add(hbox3, proportion=1, flag=wx.LEFT|wx.RIGHT|wx.EXPAND,
#             border=10)
#
#         vbox.Add((-1, 25))
#
#         hbox4 = wx.BoxSizer(wx.HORIZONTAL)
#         cb1 = wx.CheckBox(panel, label='Case Sensitive')
#         cb1.SetFont(font)
#         hbox4.Add(cb1)
#         cb2 = wx.CheckBox(panel, label='Nested Classes')
#         cb2.SetFont(font)
#         hbox4.Add(cb2, flag=wx.LEFT, border=10)
#         cb3 = wx.CheckBox(panel, label='Non-Project classes')
#         cb3.SetFont(font)
#         hbox4.Add(cb3, flag=wx.LEFT, border=10)
#         vbox.Add(hbox4, flag=wx.LEFT, border=10)
#
#         vbox.Add((-1, 25))
#
#         hbox5 = wx.BoxSizer(wx.HORIZONTAL)
#         btn1 = wx.Button(panel, label='Ok', size=(70, 30))
#         hbox5.Add(btn1)
#         btn_close = wx.Button(panel, label='Close', size=(70, 30))
#         hbox5.Add(btn_close, flag=wx.LEFT|wx.BOTTOM, border=5)
#         vbox.Add(hbox5, flag=wx.ALIGN_RIGHT|wx.RIGHT, border=10)
#
#         panel.SetSizer(vbox)


def main():
    pvc.init_pvcam()
    cam = next(PVCamera.detect_camera())

    app = wx.App()
    main_gui = Main_GUI(parent=None, title='blabla', cam=cam)
    main_gui.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()