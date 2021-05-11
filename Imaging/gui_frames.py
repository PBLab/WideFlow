import wx
from pubsub.pub import Publisher
import os
from utils.imaging_utils import load_config
wildcard = "config json (*.json)|*.json|" \
           "All files (*.*)|*.*"


class AcquisitionConfig(wx.Dialog):
    def __init__(self, config, *args, **kwargs):
        super(AcquisitionConfig, self).__init__(*args, **kwargs)
        self.configs = config
        self.currentDirectory = os.getcwd()
        self.InitUI()

    def InitUI(self):
        self.main_layout()
        self.Move((0, 0))
        self.SetSize((500, 300))
        self.SetTitle('Session Configurations')
        self.Centre()

    def main_layout(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour("gray")

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(9)

        vbox = wx.BoxSizer(wx.VERTICAL)

        txt_lbl = wx.StaticText(panel, wx.ID_ANY, "Select the Input File")
        vbox.Add(txt_lbl)

        openFileDlgBtn = wx.Button(panel, label="Browse")
        vbox.Add(openFileDlgBtn, 0, wx.ALL, 10)


        panel.SetSizer(vbox)
        self.Show()

        openFileDlgBtn.Bind(wx.EVT_BUTTON, self.onOpenFile)
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def onOpenFile(self, event):
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory,
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPaths()
            self.configs = load_config(path[0])

    def onSaveFile(self, event):
        pass

    def onClose(self, evt):
        # dlg = wx.MessageDialog(self, "Do you want to update the session configurations?", "Confirm Exit",
        #                        wx.OK | wx.CANCEL | wx.ICON_QUESTION)
        # result = dlg.ShowModal()
        # dlg.Destroy()
        # if result == wx.ID_CANCEL:
        #     event = MyCustomEvent(resultOfDialog="User Clicked CANCEL")
        #     self.GetEventHandler().ProcessEvent(event)
        # else:  # result == wx.ID_OK
        #     event = MyCustomEvent(resultOfDialog="User Clicked OK")
        #     self.GetEventHandler().ProcessEvent(event)
        self.Destroy()


def main():

    app = wx.App()
    main_gui = AcquisitionConfig(parent=None)
    main_gui.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()