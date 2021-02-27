import wx


class CameraConfig(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(CameraConfig, self).__init__(*args, **kwargs)
        self.InitUI()

    def InitUI(self):
        self.Move((0, 0))
        self.SetSize((300, 300))
        self.SetTitle('WideFlow')
        self.Centre()

    def main_layout(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour("gray")

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(9)

        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        st1 = wx.StaticText(panel, label='frame rate')
        st1.SetFont(font)
        hbox1.Add(st1, flag=wx.RIGHT, border=8)
        tc = wx.TextCtrl(panel)
        hbox1.Add(tc, proportion=1)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
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


class AcquisitionConfig(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(AcquisitionConfig, self).__init__(*args, **kwargs)
        self.InitUI()

    def InitUI(self):
        self.main_layout()
        self.Move((0, 0))
        self.SetSize((300, 300))
        self.SetTitle('WideFlow')
        self.Centre()

    def main_layout(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour("gray")

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(9)

        vbox_main = wx.BoxSizer(wx.VERTICAL)

        vbox_processes = wx.BoxSizer(wx.VERTICAL)
        hbox_title = wx.BoxSizer(wx.HORIZONTAL)
        vbox_processes.Add(hbox_title, flag=wx.LEFT, border=10)

        st1 = wx.StaticText(panel, label='Processes')
        st1.SetFont(font)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(st1)
        p1 = wx.CheckBox(panel, label='Process 1')
        p1.SetFont(font)
        hbox1.Add(p1)
        m1 = wx.CheckBox(panel, label='metric 1')
        m1.SetFont(font)
        hbox1.Add(m1)
        vbox_processes.Add(hbox1, flag=wx.LEFT, border=10)
        vbox_processes.Add((-1, 25))

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        p2 = wx.CheckBox(panel, label='Process 2')
        p2.SetFont(font)
        hbox2.Add(p2)
        m2 = wx.CheckBox(panel, label='metric 2')
        m2.SetFont(font)
        hbox2.Add(m2)
        vbox_processes.Add(hbox2, flag=wx.LEFT, border=10)
        vbox_processes.Add((-1, 25))

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        p3 = wx.CheckBox(panel, label='Process 3')
        p3.SetFont(font)
        hbox3.Add(p3)
        m3 = wx.CheckBox(panel, label='metric 3')
        m3.SetFont(font)
        hbox3.Add(m3)
        vbox_processes.Add(hbox3, flag=wx.LEFT, border=10)
        vbox_processes.Add((-1, 25))

        vbox_main.Add(vbox_processes, flag=wx.LEFT, border=10)
        vbox_main.Add((-1, 25))
        panel.SetSizer(vbox_main)


def main():

    app = wx.App()
    main_gui = AcquisitionConfig(parent=None, title='blabla')
    main_gui.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()