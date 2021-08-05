from ctypes import *
import time
import os
import sys
import platform
import tempfile
import re
if sys.version_info >= (3,0):
    import urllib.parse


ximc_dir = 'C:\\Users\\Pablo2P2\\Desktop\\libximc-2.12.1-all.tar\\ximc-2.12.1\\ximc'
ximc_package_dir = 'C:\\Users\\Pablo2P2\\Desktop\\libximc-2.12.1-all.tar\\ximc-2.12.1\\ximc\\crossplatform\\wrappers\\python'
sys.path.append(ximc_package_dir)

if platform.system() == "Windows":
    # Determining the directory with dependencies for windows depending on the bit depth.
    arch_dir = "win64" if "64" in platform.architecture()[0] else "win32" #
    libdir = os.path.join(ximc_dir, arch_dir)
    os.environ["Path"] = libdir + ";" + os.environ["Path"]  # add dll path into an environment variable

try:
    from pyximc import *
except ImportError as err:
    print("Can't import pyximc module. The most probable reason is that you changed the relative location of the testpython.py and pyximc.py files. See developers' documentation for details.")
    exit()
except OSError as err:
    # print(err.errno, err.filename, err.strerror, err.winerror) # Allows you to display detailed information by mistake.
    if platform.system() == "Windows":
        if err.winerror == 193:  # The bit depth of one of the libraries bindy.dll, libximc.dll, xiwrapper.dll does not correspond to the operating system bit.
            print("Err: The bit depth of one of the libraries bindy.dll, libximc.dll, xiwrapper.dll does not correspond to the operating system bit.")
            # print(err)
        elif err.winerror == 126:  # One of the library bindy.dll, libximc.dll, xiwrapper.dll files is missing.
            print("Err: One of the library bindy.dll, libximc.dll, xiwrapper.dll is missing.")
            # print(err)
        else:           # Other errors the value of which can be viewed in the code.
            print(err)
        print("Warning: If you are using the example as the basis for your module, make sure that the dependencies installed in the dependencies section of the example match your directory structure.")
        print("For correct work with the library you need: pyximc.py, bindy.dll, libximc.dll, xiwrapper.dll")
    else:
        print(err)
        print ("Can't load libximc library. Please add all shared libraries to the appropriate places. It is decribed in detail in developers' documentation. On Linux make sure you installed libximc-dev package.\nmake sure that the architecture of the system and the interpreter is the same")
    exit()


print("Library loaded")
sbuf = create_string_buffer(64)
lib.ximc_version(sbuf)
print("Library version: " + sbuf.raw.decode().rstrip("\0"))

# Set bindy (network) keyfile. Must be called before any call to "enumerate_devices" or "open_device" if you
# wish to use network-attached controllers. Accepts both absolute and relative paths, relative paths are resolved
# relative to the process working directory. If you do not need network devices then "set_bindy_key" is optional.
# In Python make sure to pass byte-array object to this function (b"string literal").
lib.set_bindy_key(os.path.join(ximc_dir, "win32", "keyfile.sqlite").encode("utf-8"))


class StageControl:
    def __init__(self, controller_name=None, devices_names=None, devices_id=None, engine_settings=None):
        probe_flags = EnumerateFlags.ENUMERATE_PROBE + EnumerateFlags.ENUMERATE_NETWORK
        enum_hints = b"addr=192.168.0.1,172.16.2.3"
        # enum_hints = b"addr=" # Use this hint string for broadcast enumerate
        self.devenum = lib.enumerate_devices(probe_flags, enum_hints)
        self.dev_count = lib.get_device_count(self.devenum)
        self.controller_name = controller_name or controller_name_t()
        self.devices_names = devices_names or self.get_devices_names()
        self.devices_id = devices_id or self.get_devices_ids()
        self.devices_initial_position = self.get_position()

        self.eng = engine_settings or engine_settings_t()
        self.devices_settings = []
        for dev_ind in range(0, self.dev_count):
            if isinstance(self.eng, list):
                self.devices_settings.append(lib.get_engine_settings(self.devices_id[dev_ind], byref(self.eng[dev_ind])))
            else:
                self.devices_settings.append(lib.get_engine_settings(self.devices_id[dev_ind], byref(self.eng)))

        self.devices_position = self.get_position()
        self.devices_move_settings = self.get_speed()

    def get_devices_names(self):
        devices_names = []
        for dev_ind in range(0, self.dev_count):
            device_name = lib.get_device_name(self.devenum, dev_ind)
            if type(device_name) is str:
                device_name = device_name.encode()
            devices_names.append(device_name)

        return devices_names

    def get_devices_ids(self):
        devices_id = []
        for dev_ind in range(0, self.dev_count):
             devices_id.append(lib.open_device(self.devices_names[dev_ind]))

        return devices_id

    def set_devices_settings_to_MICROSTEP_MODE_FRAC_256(self, device_id):
        eng = engine_settings_t()
        eng.MicrostepMode = MicrostepMode.MICROSTEP_MODE_FRAC_256
        lib.set_engine_settings(device_id, byref(eng))

    def set_speed(self, device_id, speed):
        # Create move settings structure
        mvst = move_settings_t()
        mvst.Speed = int(speed)
        lib.set_move_settings(device_id, byref(mvst))

    def get_speed(self):
        mvst = move_settings_t()
        speeds = []
        for dev_ind in range(0, self.dev_count):
            speeds.append(lib.get_move_settings(self.devices_id[dev_ind], byref(mvst)))
        return speeds

    def set_position(self, position_list):
        '''

        :param position_list: each element of the list include: device_id, distance, udistance
        :return:
        '''
        for dev_pos in position_list:
            self.move_device(self, self.dev_pos[0], dev_pos[1], udistance[2])

    def get_position(self):
        pos_list = []
        for dev_id in self.devices_id:
            pos = get_position_t(dev_id)
            pos_list.append([pos.Position, pos.uPosition])
        return pos_list

    def move_device(self, device_id, distance, udistance):
        lib.command_move(device_id, distance, udistance)

    def stop_stage(self):
        for dev_id in self.devices_id:
            self.stop_device(dev_id)

    def stop_device(self, device_id):
        lib.command_stop(device_id)

    def soft_stop_stage(self):
        # for device_id in self.devices_id:
        #     self.soft_stop_device(device_id)
        pass

    def soft_stop_device(self, device_id):
        # lib.command_sttp(device_id)  # TODO: figure out why it's not working
        pass

    def emergency_stop(self):
        '''
        method to implement emergency brake when motors are jammed
        '''
        pass


# stage = StageControl()
# z=3