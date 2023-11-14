import tempfile
from pathlib import Path

import streamlit as st

from talus_drc_helper.dispenser_io import (
    make_templates_from_dispenser_xml,
)

st.markdown("""
# Dispenser XML to Template Converter

## Instructions

1. Upload a dispenser XML file
2. Download the generated templates
3. Add sheets called `cell_count__time0`, this should have only two columns. `cell_line` and `cell_count__time0`
4. Add a sheet called `cell_count` to the template (first row should only be number 1->12 for a 96 well plate), first columns should only be high caps letters (top left space is empty).
5. Add a `cell_line` sheet and annotate the cell line (use the `compound` sheet as a reference)
5. Add the `user`, `protocol_id` and `plate_id` fields to the metadata sheet.
6. (ONLY BECAUSE SEBASTIAN HAS NOT IMPLEMENTED THIS YET) label with `CTRL` or `DMSO` the wells you want to use for normalization.
""")

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
