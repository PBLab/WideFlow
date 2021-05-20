import os
import wx


class ConfigurationPanel(wx.Panel):

    def __init__(self, parent, config=None):
        self.config = config
        wx.Panel.__init__(self, parent)

        # define text controller
        self.my_text = wx.TextCtrl(self, style=wx.TE_MULTILINE)

        # define buttons
        btn_load = wx.Button(self, label='Open JSON File')
        btn_load.Bind(wx.EVT_BUTTON, self.onOpen)

        btn_save = wx.Button(self, label='Save File')
        btn_save.Bind(wx.EVT_BUTTON, self.onSave)

        btn_save_as = wx.Button(self, label='Save File As')
        btn_save_as.Bind(wx.EVT_BUTTON, self.onSaveAs)

        btn_update = wx.Button(self, label='Update\nConfigurations')
        btn_update.Bind(wx.EVT_BUTTON, self.onUpdateConfig)

        # create panel layout
        btn_sizer = wx.BoxSizer(wx.VERTICAL)
        btn1 = wx.BoxSizer(wx.HORIZONTAL)
        btn1.Add(btn_load, 0, wx.ALL|wx.CENTER, 5)
        btn1.Add(btn_save, 0, wx.ALL|wx.CENTER, 5)
        btn1.Add(btn_save_as, 0, wx.ALL|wx.CENTER, 5)
        btn_sizer.Add(btn1)

        btn2 = wx.BoxSizer(wx.HORIZONTAL)
        btn2.Add(btn_update, 0, wx.ALL | wx.CENTER, 5)
        btn_sizer.Add(btn2)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.my_text, 1, wx.ALL | wx.EXPAND)
        sizer.Add(btn_sizer, 1, wx.ALL | wx.EXPAND)

        self.SetSizer(sizer)

    def onOpen(self, event):
        wildcard = "TXT files (*.json)|*.json"
        dialog = wx.FileDialog(self, "Open Text Files", wildcard=wildcard,
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if dialog.ShowModal() == wx.ID_CANCEL:
            return

        path = dialog.GetPath()
        self.file_path = path

        if os.path.exists(path):
            with open(path) as fobj:
                for line in fobj:
                    self.my_text.WriteText(line)

        self.config = self.my_text.Value

    def onSaveAs(self, event):
        with wx.FileDialog(self, "Save as", wildcard="TXT files (*.json)|*.json",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # save the current contents in the file
            pathname = fileDialog.GetPath()
            try:
                with open(pathname, 'w') as file:
                    self.my_text.SaveFile(pathname)
            except IOError:
                wx.LogError("Cannot save current data in file '%s'." % pathname)
            self.file_path = pathname

    def onSave(self, event):
        try:
            with open(self.file_path, 'w') as file:
                self.my_text.SaveFile(self.file_path)
        except IOError:
            wx.LogError("Cannot save current data in file '%s'." % self.file_path)

    def onUpdateConfig(self, event):
        self.config = self.my_text.Value


class ConfigurationWizard(wx.Frame):

    def __init__(self, config=None):
        wx.Frame.__init__(self, None, title='Session Acquisition Configurations')
        self.panel = ConfigurationPanel(self, config=config)
        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.Show()

    def onClose(self, event):
        #TODO: return configuration value back to main GUI frame
        config = self.panel.config
        self.Close()


if __name__ == '__main__':
    app = wx.App(False)
    frame = ConfigurationWizard()
    app.MainLoop()




