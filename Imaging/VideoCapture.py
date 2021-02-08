import wx
import cv2


class VideoCapture(wx.Frame):
    def __init__(self, capture, fps, *args, **kwargs):
        super(VideoCapture, self).__init__(*args, **kwargs)
        self.InitUI()
        self.capture = capture
        ret, frame = self.capture.read()
        height, width = frame.shape[:2]
        self.bmp = wx.BitmapFromBuffer(width, height, frame)

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
        ret, frame = self.capture.read()
        if ret:
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()
