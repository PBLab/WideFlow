import h5py
from functools import reduce
import operator


def decompose_dict_to_h5_groups(f, d, grp='/'):
    """

    Args:
        f: h5py File object
        d: dict
        grp: base group

    Returns:

    """
    for key, val in d.items():
        if isinstance(val, dict):
            grp_new = grp + key + '/'
            f.create_group(grp_new)
            decompose_dict_to_h5_groups(f, val, grp_new)
        else:
            f.create_dataset(grp + key, data=val)


def decompose_h5_groups_to_dict(f, d, grp='/'):
    """

    Args:
        f: h5py File object
        d: dict
        grp: base group

    Returns:

    """
    def recursion_func(fr, dr, base_grp='/', grpr='/'):
        n = len(base_grp) - 1
        for key, val in fr[grpr].items():
            new_grp = grpr + key + '/'
            grp_keys = new_grp[n:].split('/')[1:-1]
            if type(fr[new_grp]) is h5py.Group:
                set_in_dict(dr, grp_keys, {})
                recursion_func(fr, dr, base_grp, new_grp)
            else:
                set_in_dict(d, grp_keys, val[()])

    recursion_func(f, d, grp, grp)


def get_from_dict(data_dict, map_list):
    return reduce(operator.getitem, map_list, data_dict)


def set_in_dict(data_dict, map_list, value):
    get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = value
