import re

import streamlit as st
import shutil
from pathlib import Path


def setup(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


def set_value(k, v):
    st.session_state[k] = v


def safe_move(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(exist_ok=True, parents=True)
    shutil.move(src, dst)


def parse_mask_name(fn):
    return re.findall(r"mask_([\w\d]+)_\d+.png", fn)[0]


def parse_mask_val(fn):
    return int(re.findall(r"mask_[\w\d]+_(\d+).png", fn)[0])
