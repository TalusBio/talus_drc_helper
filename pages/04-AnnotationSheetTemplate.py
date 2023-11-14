import tempfile
from pathlib import Path

import streamlit as st

from talus_drc_helper.dispenser_io import (
    make_templates_from_dispenser_xml,
)

with tempfile.TemporaryDirectory() as ASDASDASD:
    st.markdown("## Input Data\n")
    infile = st.file_uploader("Input file to use", accept_multiple_files=False)
    if infile is None:
        st.warning("No files uploaded")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
        tmp.write(infile.getvalue())
        with st.spinner("Generating the files"):
            make_templates_from_dispenser_xml(tmp.name, Path(ASDASDASD))

        st.warning("DONE!")

    st.markdown("## Download Plots\n")
    for x in Path(ASDASDASD).rglob("*.html"):
        with open(x, "rb") as f:
            data = f.read()
            st.download_button(x.name, data, file_name=x.name)

    st.markdown("## Download Templates\n")
    for x in Path(ASDASDASD).rglob("*.xlsx"):
        with open(x, "rb") as f:
            data = f.read()
            st.download_button(x.name, data, file_name=x.name)
