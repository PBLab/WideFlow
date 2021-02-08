import wx
import cv2
import numpy as np


class VideoCapture(wx.Frame):
    def __init__(self, cam, *args, **kwargs):
        super(VideoCapture, self).__init__(*args, **kwargs)
        self.InitUI()

        self.cam = cam
        self.cam.open()
        frame = self.cam.get_frame()
        height, width = frame.shape[:2]

        frame = np.stack((frame, frame, frame), axis=2)
        self.bmp = wx.Bitmap.FromBuffer(width, height, frame)

        fps = 20
        self.timer = wx.Timer(self)
        self.timer.Start(1000. / fps)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)

    def InitUI(self):

        self.Move((0, 0))
        self.SetSize((300, 300))
        self.SetTitle('WideFlow')
        self.Centre()

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def NextFrame(self, event):
        frame = self.cam.get_frame()
        frame = np.stack((frame, frame, frame), axis=2)
        self.bmp.CopyFromBuffer(frame)


