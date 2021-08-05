import wx
import cv2
import numpy as np


class VideoCapture(wx.Frame):
    def __init__(self, cam, *args, **kwargs):
        super(VideoCapture, self).__init__(*args, **kwargs)
        self.cam = cam
        self.cam.open()
        # TODO: height and width determined by frames dims, or cam.shape
        self.height = cam.sensor_size[0]
        self.width = cam.sensor_size[1]
        # weight and height should be driven from camera shape instead the sensor size
        self.imageBit = wx.Bitmap(wx.Image(self.height, self.width))
        self.frame = np.array(self.cam.get_frame(), dtype=np.float32)
        # self.frame = np.zeros((self.height, self.width, 3))
        self.bmp = wx.Bitmap.FromBuffer(self.width, self.height, self.frame)

        self.InitUI()

        if cam.exp_time == 0:
            fps = 25
        else:
            fps = 1 / cam.exp_time
        self.timer = wx.Timer(self)
        self.timer.Start(1. / fps)

    def InitUI(self):

        self.main_layout()
        self.Move((0, 0))
        self.SetSize((600, 600))
        self.SetTitle('live')
        self.Centre()

    def main_layout(self):
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour("gray")
        # self.SetSize(self.height+100, self.width)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        btn_play = wx.Button(self.panel, label='play', size=(70, 30))
        self.btn_play = btn_play
        hbox.Add(btn_play)
        vbox.Add(hbox, flag=wx.ALIGN_TOP, border=10)
        vbox.Add((-1, 25))

        self.staticBit = wx.StaticBitmap(self.panel, wx.ID_ANY, self.imageBit)
        self.staticBit.SetPosition((0, 35))
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.staticBit, border=10)
        vbox.Add(hbox2)
        vbox.Add((-1, 25))

        self.Bind(wx.EVT_BUTTON, self.play, btn_play)
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def NextFrame(self, event):
        # frame = self.cam.get_frame()
        # self.frame = np.stack((frame, frame, frame), axis=2)
        self.frame = np.array(self.cam.get_live_frame(), dtype=np.float32)
        self.bmp.CopyFromBuffer(self.frame)
        self.staticBit.SetBitmap(self.bmp)
        self.Refresh()

    def onClose(self, event):
        self.cam.stop_live()
        self.cam.close()
        self.Unbind(wx.EVT_TIMER)
        self.Destroy()

    def play(self, event):
        self.cam.start_live()
        self.Bind(wx.EVT_TIMER, self.NextFrame)

        self.Bind(wx.EVT_BUTTON, self.pause, self.btn_play)
        self.btn_play.SetLabel('pause')

    def pause(self, event):
        self.cam.stop_live()
        self.Unbind(wx.EVT_TIMER)

        self.Bind(wx.EVT_BUTTON, self.play, self.btn_play)
        self.btn_play.SetLabel('play')