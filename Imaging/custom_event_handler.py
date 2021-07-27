import wx
import wx.lib.newevent

(MyCustomEvent, EVT_CUSTOM) = wx.lib.newevent.NewEvent()


class CustomEventTracker(wx.EvtHandler):
    def __init__(self, log, processingCodeFunctionHandle):
        wx.EvtHandler.__init__(self)
        self.processingCodeFunctionHandle = processingCodeFunctionHandle
        EVT_CUSTOM(self, self.MyCustomEventHandler)

    def MyCustomEventHandler(self, event):
        self.processingCodeFunctionHandle(event.resultOfDialog)
        event.Skip()