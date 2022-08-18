import os
import streamlit as st
import wx
from pathlib import Path as P

from tqdm import tqdm

from brainseg.streamlit.manager import get_list_mask, change_priority, has_zero, reset, flush_out
from brainseg.streamlit.load import load_mask, load_image, load_superpose_mask, load_multiply_mask, has_lowres, \
    load_lowres, load_superpose_mask_3
from brainseg.streamlit.utils import setup, set_value, parse_mask_name


def initialization():
    # Initialization
    st.set_page_config(layout="wide")
    setup('folder_path', "/media/nathan/LaCie/Data/multi_bires_x8_224_v2")
    setup('current', None)
    setup('current_mask', None)
    setup('filelist', None)
    setup('history', [])
    load_list()

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


def list_files(folder_path, filelist=None):
    if filelist is None:
        filelist = []

        for f in os.listdir(folder_path):
            if (P(folder_path) / f).is_dir():
                filelist.append(f)

    rad = st.radio("Select the file", filelist)

    for f in filelist:
        if rad == f:
            set_value("current", f)


def load_list():
    max_size = 100
    ls = []
    folder_path = st.session_state.folder_path
    if st.session_state.filelist is not None:
        pass

    if folder_path is None:
        return

    count = 0
    for f in os.listdir(folder_path):
        if not (P(folder_path) / f).is_dir() or f.startswith("."):
            continue

        # keep only if has a zero val
        if has_zero(folder_path, f):
            ls.append(f)
            count += 1

        if count >= max_size:
            break

    st.session_state.filelist = ls


def undo():
    last_source, last_target = st.session_state.history.pop()
    reset(last_source, last_target)


def empty():
    flush_out(st.session_state.folder_path, hard=True)


def sidebar():
    with st.sidebar:
        button = st.button("Change root directory")

        if button:
            dlg = wx.DirDialog(None, "Choose directory", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:
                st.session_state.folder_path = dlg.GetPath()
                st.session_state.current = None
                load_list()

        # If unchecked then shows only the 000s
        ch = st.checkbox("Show all")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("Refresh", on_click=load_list)

        with c2:
            if len(st.session_state.history) > 0:
                st.button("Undo", on_click=undo)

        with c3:
            st.button("Empty", on_click=empty)

        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
                    unsafe_allow_html=True)

        list_ids = ["none", "bad image", "bad mask", "good mask", "very mask"]
        show_id = st.selectbox("Select an id", list_ids, index=0)

        st.header("Current folder", st.session_state.folder_path)
        if st.session_state.folder_path is not None:
            st.write(st.session_state.folder_path)
            filelist = st.session_state.filelist if not ch else None
            list_files(st.session_state.folder_path,
                       filelist=filelist)

        return show_id


def draw_mask(path, name, mask_name, c1, c2):
    with c1:
        st.image(load_image(path, name), width=448)
        st.image(load_superpose_mask_3(path, name, mask_name), width=448)

    with c2:
        st.image(load_mask(path, name, mask_name), width=448)
        # here if lowres then show otherwise multiply
        if has_lowres(path, name):
            img = load_lowres(path, name)
            st.image(img, width=448)


def change_mask_priority(priority):
    source, target = change_priority(
        st.session_state.folder_path, st.session_state.current,
        parse_mask_name(st.session_state.current_mask), int(priority)
    )

    st.session_state.history.append((source, target))


def build_main(ids):
    print(ids)
    # Main
    if st.session_state.current is not None:
        list_mask = get_list_mask(st.session_state.folder_path, st.session_state.current)
        sel = st.selectbox("Select a mask", list_mask, index=0)

        col_button, col_mask1, col_mask2 = st.columns([1, 4, 4])

        for mask in list_mask:
            if sel == mask:
                draw_mask(st.session_state.folder_path, st.session_state.current, mask, col_mask1, col_mask2)
                st.session_state.current_mask = mask

        with col_button:
            for i in range(20):
                st.write("")
            vals = ["000", "001", "002", "010", "020"]
            names = ["none", "bad image", "bad mask", "good mask", "very mask"]
            for i in range(len(vals)):
                b = st.button(names[i], on_click=change_mask_priority, args=(vals[i],))
                if b:
                    pass


app = wx.App()
initialization()
x = sidebar()
build_main(x)
