import re


def get_slice_number(fn):
    return re.findall(r"\D+\d+\D+(\d+)\D", fn)[0]


def get_neuron_cat(n):
    return re.findall(r"Point Cat(\d)", n)[0]
