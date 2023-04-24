from pathlib import Path

import pandas as pd
import streamlit as st

from talus_drc_helper.plate_annotations import (
    AnnotatedPlate,
    AnnotatedPlateSet,
    PlateAnnotation,
)
from talus_drc_helper.plate_io import read_dispenser_excel, read_synergy_str
from talus_drc_helper.stramlit import fit_and_show_ann_plate


def get_dispenser_plate_annotations():
    input_data = st.file_uploader("Upload a Dispenser file", type=["xlsx", "txt"])
    if input_data is None:
        return None
    dispenser_df = read_dispenser_excel(input_data)
    annotation_set = AnnotatedPlateSet.from_dispenser_df(dispenser_df)
    st.dataframe(dispenser_df)
    return annotation_set


def import_viability_files():
    infiles = st.file_uploader(
        "Synergy .txt intensity files to use:",
        accept_multiple_files=True,
        key="viability",
    )
    out = {}
    for f in infiles:
        k = f.name
        v = read_synergy_str(f.getvalue().decode("utf-8").split("\n"))
        pa = PlateAnnotation(v, "Intensity")
        out[k] = pa

    return out


def annotate_plate_dict_from_name(
    plate_dict: dict[str:PlateAnnotation], plate_ids: list[str]
):
    first_filename = Path(list(plate_dict.keys())[0]).stem
    suggested_vals = ",".join(
        [f"Field_{i+1}" for i, _ in enumerate(first_filename.split("_"))]
    )
    annotation_levels = st.text_input(
        "Annotation levels (comma separated)", value=suggested_vals
    ).split(",")
    annotation_levels = [x.strip() for x in annotation_levels]
    split_string = st.text_input(
        "Split string", value="_", help="String to split the file name on"
    )
    if "PlateID" not in annotation_levels:
        st.warning(
            "'PlateID' can be added as an annotation level, must be filled manually"
            " otherwise"
        )
        manual_id = True
        annotation_levels.append("PlateID")
    else:
        manual_id = False

    out = {}
    for k, v in plate_dict.items():
        k = Path(k).stem
        split = k.split(split_string)
        if manual_id:
            st.subheader(f"Manual Annotation from Plate {k}")
            id = st.selectbox("Plate ID", plate_ids, key=k)
            split.append(id)
        if len(split) != len(annotation_levels):
            st.warning(f"Could not annotate {k}")
            st.stop()
        out[k] = AnnotatedPlate(
            [v], plate_level_annotations=dict(zip(annotation_levels, split))
        )
    st.dataframe(pd.DataFrame([x.plate_level_annotations for x in out.values()]))
    return out


def annotate_plate_dict_from_input(
    plate_dict: dict[str:PlateAnnotation], plate_ids: list[str]
) -> dict[str, AnnotatedPlate]:
    annotation_levels = st.text_input(
        "Annotation levels (comma separated)", value="CellLine"
    ).split(",")
    annotation_levels = [x.strip() for x in annotation_levels]
    out = {}
    if "PlateID" not in annotation_levels:
        st.warning("PlateID not in annotation levels, Will be added automatically")
        annotation_levels.append("PlateID")

    for k, v in plate_dict.items():
        annotations = []
        st.subheader(f"Manual Annotation from Plate {k}")
        for level in annotation_levels:
            if level == "PlateID":
                annotations.append(
                    st.selectbox(f"Plate ID for {level}", plate_ids, key=k)
                )
            else:
                annotations.append(
                    st.text_input(f"Annotation for {level}", value=level)
                )
        out[k] = AnnotatedPlate(
            [v], plate_level_annotations=dict(zip(annotation_levels, annotations))
        )
    return out


def combine_annotation_and_plate(
    plate_dict: dict[str, AnnotatedPlate], dispenser_annotations: AnnotatedPlateSet
) -> AnnotatedPlateSet:
    plate_id_type = type(list(dispenser_annotations.plates)[0])
    plate_values = {}
    for k, plate in plate_dict.items():
        plate.plate_level_annotations["PlateID"] = plate_id_type(
            plate.plate_level_annotations["PlateID"]
        )
        plate_id = plate.plate_level_annotations["PlateID"]
        if plate_id not in dispenser_annotations.plates:
            st.warning(f"Plate {plate_id} not in dispenser annotations")
            st.stop()

        plate.extend(dispenser_annotations[plate_id].annotation)
        plate_values[k] = plate

    return AnnotatedPlateSet(plate_values)


# def get_viability_data(plate_id, st_element_id, annotation_level):
#     plate_annot_level = st.text_input(
#         f"Annotation for plate {plate_id}", value=plate_id
#     )
#     infile = st.file_uploader(
#         "Input file to use", accept_multiple_files=False, key=st_element_id
#     )
#     if infile is None:
#         return None
#     input_data = read_synergy_str(infile.getvalue().decode("utf-8").split("\n"))
#     pa = PlateAnnotation(input_data, "Intensity")
#     pa = AnnotatedPlate(
#         [pa], plate_level_annotations={annotation_level: plate_annot_level}
#     )
#     return pa


############===========Start of sidebar=================############
with st.sidebar:
    # Read Dispense info
    st.title("Dispenser Annotation")
    st.markdown(
        "This page fits values from one single dispenser report (xlsx) and one or more"
        " synergy output files (.txt)"
    )
    dispenser_annotations = get_dispenser_plate_annotations()
    if dispenser_annotations is None:
        st.warning("No dispenser file uploaded")
        st.stop()

    intensity_files = import_viability_files()
    if len(intensity_files) > 0:
        info_from_name = st.checkbox("Extract annotation from file name??")
        plate_keys = list(dispenser_annotations.plates.keys())
        if info_from_name:
            intensity_files = annotate_plate_dict_from_name(intensity_files, plate_keys)
        else:
            intensity_files = annotate_plate_dict_from_input(
                intensity_files, plate_keys
            )
    else:
        st.warning("No intensity files uploaded")
        st.stop()

    plate_values = combine_annotation_and_plate(intensity_files, dispenser_annotations)


############===========End of sidebar=================############

if dispenser_annotations is not None:
    st.subheader("Annotated Data")

    ann_set = {}
    for k, v in plate_values.items():
        if v is None:
            continue
        ann_set[k] = v
        with st.expander(f"Plate {k}"):
            st.subheader(f"Plate {k}")
            st.altair_chart(v.plot(), use_container_width=False)

    if len(ann_set) == 0:
        st.subheader("No data to plot")
        st.stop()

    ann_plate = AnnotatedPlateSet(ann_set)
    fit_and_show_ann_plate(ann_plate)
