import os
import streamlit as st
import time
import wx
from pathlib import Path as P

from brainseg.streamlit.manager import get_list_mask, load_mask, load_image, \
    load_superpose_mask, load_multiply_mask, change_priority, parse_mask_name


def setup(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


def set_value(k, v):
    st.session_state[k] = v


def initialization():
    # Initialization
    st.set_page_config(layout="wide")
    setup('folder_path', "/media/nathan/LaCie/Data/whitematter_curated2")
    setup('current', None)
    setup('current_mask', None)

    st.markdown(
        """
        <style>
        .css-fg4pbf {
            background-color: #CCCCFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def list_files(folder_path):
    filelist = []

    for f in os.listdir(folder_path):
        if (P(folder_path) / f).is_dir():
            filelist.append(f)

    rad = st.radio("Select the file", filelist)

    for f in filelist:
        if rad == f:
            set_value("current", f)


def sidebar():
    with st.sidebar:
        button = st.button("Change root directory")

        if button:
            dlg = wx.DirDialog(None, "Choose directory", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:
                st.session_state.folder_path = dlg.GetPath()
                st.session_state.current = None

        st.header("Current folder", st.session_state.folder_path)
        if st.session_state.folder_path is not None:
            st.write(st.session_state.folder_path)
            list_files(st.session_state.folder_path)


def draw_mask(path, name, mask_name, c1, c2):
    with c1:
        st.image(load_image(path, name), width=448)
        st.image(load_superpose_mask(path, name, mask_name), width=448)

    with c2:
        st.image(load_mask(path, name, mask_name), width=448)
        st.image(load_multiply_mask(path, name, mask_name), width=448)


app = wx.App()
initialization()
sidebar()


# Main
if st.session_state.current is not None:
    list_mask = get_list_mask(st.session_state.folder_path, st.session_state.current)
    sel = st.selectbox("Select a mask", list_mask, index=0)  # todo change the index depending on the selected mask

    col_mask1, col_mask2, col_button = st.columns([3, 3, 1])

    for mask in list_mask:
        if sel == mask:
            draw_mask(st.session_state.folder_path, st.session_state.current, mask, col_mask1, col_mask2)
            st.session_state.current_mask = mask

    with col_button:
        vals = ["000", "001", "010", "020", "100"]
        for i in range(len(vals)):
            b = st.button(vals[i], on_click=lambda x: change_priority(
                st.session_state.folder_path, st.session_state.current,
                parse_mask_name(st.session_state.current_mask), int(x)
            ), args=(vals[i],))
            if b:
                pass

