"""
@file opencl_explore.py

Explore OpenCL platforms and devices.
"""
import pyopencl as cl


def size_human(s):
    """Converts size in Bytes into human readable size"""
    c = ['b', 'Kb', 'Mb', 'Gb']
    i = 0
    while s >= 1024:
        s /= 1024
        i += 1
    return str(int(s)) + c[i]


def get_defaults(device_type=cl.device_type.GPU):
    """Get OpenCL default indices (GPU with largest global memory)"""
    default_platform, default_device = 0, 0
    default_device_mem_size = 0
    try:
        for i, plt in enumerate(cl.get_platforms()):
            for j, dev in enumerate(plt.get_devices()):
                if dev.type == device_type and \
                        default_device_mem_size < dev.global_mem_size:
                    default_device = j
                    default_platform = i
    except cl.LogicError:
        import sys
        sys.stderr.write("WARNING : You are running for a non-OpenCL place " +
                         "such as a frontend. Using a default id: 0 0")
    return default_platform, default_device


def explore(device_type=cl.device_type.GPU):
    """Explore platforms and devices"""
    p_data, d_data = {}, {}
    try:
        platforms = cl.get_platforms()
        platforms_info = ["name    ",
                          "version ",
                          "vendor  "]
        all_devices = []
        unchanged = lambda x: x
        boolean = lambda b: 'Yes' if b else 'No'
        type_str = cl.device_type.to_string
        devices_info = [("name                    ", unchanged),
                        ("version                 ", unchanged),
                        ("vendor                  ", unchanged),
                        ("profile                 ", unchanged),
                        ("type                    ", type_str),
                        ("compiler_available      ", boolean),
                        ("double_fp_config        ", boolean),
                        ("single_fp_config        ", boolean),
                        ("global_mem_size         ", size_human),
                        ("local_mem_size          ", size_human),
                        ("max_compute_units       ", unchanged),
                        ("max_mem_alloc_size      ", size_human),
                        ("max_work_group_size     ", unchanged),
                        ("max_work_item_dimensions", unchanged),
                        ("max_work_item_sizes     ", unchanged),
                        ("extensions              ", unchanged),
                         ]
        p_str_max = []
        d_str_max = {}
        out = ""
        for i, plt in enumerate(platforms):
            p_str_max.append(0)
            devices = plt.get_devices()
            all_devices += devices
            p_data[plt] = []
            for plt_info in platforms_info:
                p_data[plt].append(eval("plt." + plt_info))
                if len(p_data[plt][-1]) > p_str_max[i]:
                    p_str_max[i] = len(p_data[plt][-1])
            for j, dev in enumerate(devices):
                d_str_max[dev] = 0
                d_data[dev] = []
                for dev_info in devices_info:
                    d_data[dev].append(
                        dev_info[1](eval("dev." + dev_info[0])))
                    if dev_info != devices_info[-1] and \
                            len(str(d_data[dev][-1])) > d_str_max[dev]:
                        d_str_max[dev] = len(str(d_data[dev][-1]))
                # platform index
                d_data[dev].append(i)
        default_platform, default_device = \
            get_defaults(device_type=device_type)

        out += "Platforms informations:\n  Id       |"
        for i, plt in enumerate(platforms):
            out += str(i) + ' ' * (p_str_max[i] - len(str(i))) + ' |'
        for i, plt_info in enumerate(platforms_info):
            out += "\n  " + plt_info + " |"
            for i_p, plt in enumerate(platforms):
                out += p_data[plt][i]
                out += ' ' * (p_str_max[i_p] - len(p_data[plt][i])) + ' |'
        out += "\nDevices informations: \n  Default device           |"
        for i, dev in enumerate(all_devices):
            if i == default_device and d_data[dev][-1] == default_platform:
                out += "DEFAULT" + ' ' * (d_str_max[dev] - 7) + ' |'
            else:
                out += ' ' * (d_str_max[dev]) + ' |'
        out += "\n  Platform Id              |"
        for i, dev in enumerate(all_devices):
            out += str(d_data[dev][-1])
            out += ' ' * (d_str_max[dev] - len(str(d_data[dev][-1]))) + ' |'
        out += "\n  Id                       |"
        for i, dev in enumerate(all_devices):
            out += str(i) + ' ' * (d_str_max[dev] - len(str(i))) + ' |'

        for i, dev_info in enumerate(devices_info[:-1]):
            out += "\n  " + dev_info[0] + " |"
            for i_d, dev in enumerate(all_devices):
                out += str(d_data[dev][i])
                out += ' ' * (d_str_max[dev] - len(str(d_data[dev][i]))) + ' |'
        out += "\n"
        print (out)
    except cl.LogicError:
        pass


if __name__ == "__main__":
    import sys
    if "EXPLORE" in sys.argv:
        if "CPU" in sys.argv:
            explore(device_type=cl.device_type.CPU)
        else:
            explore()
    else:
        if "CPU" in sys.argv:
            p_id, d_id = get_defaults(device_type=cl.device_type.CPU)
        else:
            p_id, d_id = get_defaults()
        print (str(p_id) + ' ' + str(d_id))
