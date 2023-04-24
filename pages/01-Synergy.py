from __future__ import annotations

import pandas as pd
import streamlit as st

from talus_drc_helper.plate_annotations import (
    AnnotatedPlate,
    AnnotatedPlateSet,
    PlateAnnotation,
)
from talus_drc_helper.plate_io import read_synergy_str
from talus_drc_helper.stramlit import fit_and_show_ann_plate

PLATE_SIZES = ["96_well", "384_well"]


def dilution_annotation(id, plate_size):
    # Remove edges
    edge_remove = st.checkbox("Remove Edges?", value=True, key=f"edge_remove-{id}")

    # Dilution Order
    dmso_position = st.selectbox(
        "Position of the DMSO", ["first", "last"], key=f"dmso_position-{id}"
    )

    dilution_direction_aliases = {
        "Top -> Bottom": "TB",
        "Bottom -> Top": "BT",
        "Left -> Right": "LR",
        "Left <- Center -> Right": "CLR",
        "Left -> Center <- Right": "LRC",
    }
    dilution_direction = dilution_direction_aliases[
        st.selectbox(
            "What is the dilution layout?",
            options=list(dilution_direction_aliases),
            key=f"dilution_direction-{id}",
        )
    ]

    # Dilution Factor
    dilution_factor = st.number_input(
        "Dilution factor used for the serial dilution",
        min_value=1.1,
        max_value=100.0,
        step=0.01,
        key=f"dilution_factor-{id}",
    )

    # Initial Conc
    initial_conc = st.number_input(
        "Initial Concentration",
        0.1,
        1000.0,
        value=100.0,
        step=0.1,
        key=f"initial_conc-{id}",
    )

    dilution_plate = PlateAnnotation.from_dilution(
        dilution_factor,
        initial_dose=initial_conc,
        zero_position=dmso_position,
        direction=dilution_direction,
        skip_edges=edge_remove,
        plate_size=plate_size,
    )
    return dilution_plate


def block_annotation(id, plate_size):
    edge_remove = st.checkbox("Remove Edges?", value=True, key=f"edge_remove-{id}")
    num_reps = st.slider(
        label="Number of replicates",
        min_value=1,
        max_value=8,
        value=3,
        key=f"num_reps-{id}",
    )
    default_names = "\n".join([f"Drug{x}" for x in range(7)])
    group_names = st.text_area(
        label="Group Names", value=default_names, key=f"group_name_id-{id}"
    )
    group_names = [x.strip() for x in group_names.split("\n")]
    direction = st.selectbox(
        "What is the layout of the blocks?",
        options=["Vertical", "Horizontal"],
        key=f"direction-{id}",
    )

    replicates_plate = PlateAnnotation.from_replicates(
        num_reps,
        labels=group_names,
        layout=direction,
        plate_size=plate_size,
        skip_edges=edge_remove,
    )
    return replicates_plate


def add_annotation_layer(id: str, plate_size: str) -> AnnotatedPlate | None:
    use = st.checkbox(f"Use Annotation Layer {id}", value=id == 0, key=f"checkbox-{id}")
    if not use:
        return None
    with st.container():
        annotation_type = st.selectbox(
            "What type of annotation?",
            options=["Dilution", "Replicates"],
            key=f"selectbox-{id}",
        )

        annotation_name = st.text_input(
            "New Annotation Name",
            value="Concentration" if annotation_type == "Dilution" else "Drug",
            key=f"textinput-{id}",
        )

        if annotation_type == "Dilution":
            dilution_plate = dilution_annotation(id, plate_size=plate_size)
            dilution_plate.annotation_name = annotation_name
            return dilution_plate
        elif annotation_type == "Replicates":
            replicates_plate = block_annotation(id, plate_size=plate_size)
            replicates_plate.annotation_name = annotation_name
            return replicates_plate


def get_viability_data() -> AnnotatedPlateSet:
    st.markdown(
        "## Input Data\nThe input data is one or more .txt files from the synergy"
        " reader."
    )
    plate_annot_level = st.text_input("Plate level annotation name:", value="Cell Line")
    infiles = st.file_uploader("Input file to use", accept_multiple_files=True)
    vals = [st.text_input(f"**Label for {f.name}**") for i, f in enumerate(infiles)]
    if len(infiles) == 0:
        st.warning("No files uploaded")
        st.stop()
    input_data = {
        k: read_synergy_str(v.getvalue().decode("utf-8").split("\n"))
        for k, v in zip(vals, infiles)
    }
    plate_data = {}
    for (k, v), file in zip(input_data.items(), infiles):
        nv = AnnotatedPlate(
            [PlateAnnotation(v, "Intensity")],
            plate_level_annotations={plate_annot_level: k},
        )
        plate_data[file.name] = nv

    input_data = AnnotatedPlateSet(plate_data)
    return input_data


with st.sidebar:
    # File input
    plate_size = st.selectbox(
        "Plate Size",
        options=PLATE_SIZES,
        key="plate_size",
    )
    input_data = get_viability_data()
    extra_annotations = {}

    # Add extra annotations
    st.markdown(
        "## Optional Annotation Layers\nThese layers can be used to add additional"
        " information to the data. \n> Please make sure at least one annotation is the"
        " concentratio."
    )
    for i in range(3):
        st.subheader(f"Optional Annotation layer {i+1}")
        ann = add_annotation_layer(id=i, plate_size=plate_size)
        if ann is not None:
            input_data.append(plate_annotation=ann)
            extra_annotations[ann.annotation_name] = ann


st.subheader("Input Data")

for k, v in input_data.items():
    st.altair_chart(
        v.annotation[0].plot().properties(title=k), use_container_width=True
    )

for k, v in extra_annotations.items():
    with st.expander(k):
        st.altair_chart(v.plot(), use_container_width=True)
        st.dataframe(v.as_long_df(), use_container_width=True)

annotations = AnnotatedPlate(annotation=list(extra_annotations.values()))
st.download_button(
    label="📥 Download Annotations as excel file!",
    data=annotations.to_excel_bytes(),
    file_name="df_test.xlsx",
)

# Replace infinitely low values with the lowest value
pd.set_option("mode.use_inf_as_na", True)
fit_and_show_ann_plate(input_data)
