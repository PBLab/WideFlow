import wx


class AquisitionConfig(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(AquisitionConfig, self).__init__(*args, **kwargs)
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
