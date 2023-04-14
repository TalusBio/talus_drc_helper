import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from talus_drc_helper.fit import DRCEstimator
from talus_drc_helper.plate_annotations import AnnotatedPlate, PlateAnnotation
from talus_drc_helper.plate_io import read_synergy_str

with st.sidebar:
    # File input
    infiles = st.file_uploader("Input file to use", accept_multiple_files=True)

    vals = [st.text_input(f"**Label for {f.name}**") for i, f in enumerate(infiles)]

    # Dilution Factor
    dilution_factor = st.number_input(
        "Dilution factor used for the serial dilution",
        min_value=1.1,
        max_value=100.0,
        step=0.01,
    )

    # Initial Conc
    initial_conc = st.number_input(
        "Initial Concentration", 0.1, 1000.0, value=100.0, step=0.1
    )

    # Dilution Order
    dmso_position = st.selectbox("Position of the DMSO", ["first", "last"])

    dilution_direction_aliases = {
        "Top -> Bottom": "TB",
        "Bottom -> Top": "BT",
        "Left -> Right": "LR",
        "Left <- Center -> Right": "CLR",
        "Left -> Center <- Right": "LRC",
    }
    dilution_direction = dilution_direction_aliases[
        st.selectbox(
            "What is the dilution layout?", options=list(dilution_direction_aliases)
        )
    ]
    opposite_layout = {
        "TB": "Vertical",
        "BT": "Vertical",
        "LR": "Horizontal",
        "CLR": "Horizontal",
        "LRC": "Horizontal",
    }[dilution_direction]

    # Remove edges
    edge_remove = st.checkbox("Remove Edges?")

    # Grouping Names
    default_names = "\n".join([f"Drug{x}" for x in range(7)])
    group_names = [
        x.strip()
        for x in st.text_area(label="Group Names", value=default_names).split("\n")
    ]
    num_reps = st.slider(
        label="Number of replicates", min_value=1, max_value=8, value=3
    )

st.subheader("Input Data")
dilution_plate = PlateAnnotation.from_dilution(
    dilution_factor,
    initial_dose=initial_conc,
    zero_position=dmso_position,
    direction=dilution_direction,
    skip_edges=edge_remove,
    plate_size="384_well",
)
input_data = {
    k: read_synergy_str(v.getvalue().decode("utf-8").split("\n"))
    for k, v in zip(vals, infiles)
}
input_data = {k: PlateAnnotation(v, "Intensity") for k, v in input_data.items()}

for k, v in input_data.items():
    st.altair_chart(v.plot().properties(title=k), use_container_width=True)


st.subheader("Replicate Annotations")
replicates_plate = PlateAnnotation.from_replicates(
    num_reps, labels=group_names, layout=opposite_layout
)
st.altair_chart(
    replicates_plate.plot().properties(title="Replicates"), use_container_width=True
)

st.subheader("Dilution Series")
st.altair_chart(
    dilution_plate.plot().properties(title="Dilutiuon Series"), use_container_width=True
)

st.subheader("Annotated Data")
ann_df = []
for k, v in input_data.items():
    ann_plate = AnnotatedPlate([dilution_plate, replicates_plate, v])
    ann_plate_df = ann_plate.as_df()
    ann_plate_df["PlateLabel"] = k
    ann_df.append(ann_plate_df)

ann_df = pd.concat(ann_df).replace(r"^\s*$", np.nan, regex=True).dropna()
ann_df = ann_df.astype({"Dose": float, "Intensity": float})
ann_df["LogDose"] = np.log10(ann_df["Dose"])

# Replace infinitely low values with the lowest value
pd.set_option("mode.use_inf_as_na", True)
replacement = ann_df["LogDose"].dropna().min() - 1

st.warning(f"Replacement value: {replacement}")

ann_df["LogDose"] = ann_df["LogDose"].fillna(replacement)
ann_df["Replicates"] = ann_df["Replicates"].astype("category")
ann_df["PlateLabel"] = ann_df["PlateLabel"].astype("category")

st.dataframe(ann_df, use_container_width=True)

st.subheader("DRC Fitting")
drcs = {}
for i, x in ann_df.groupby(["PlateLabel", "Replicates"]):
    try:
        drcs[i] = DRCEstimator().fit(X=x["LogDose"], y=x["Intensity"])
    except RuntimeError as e:
        st.warning(f"Failed to fit {i}: {e}")


plot_chart = (
    alt.Chart(ann_df)
    .mark_circle(size=20)
    .encode(
        alt.X("LogDose:Q", sort="ascending", title="Log Dose"),
        # x="Dose",
        alt.Y("Intensity"),
        color="Replicates",
    )
)

final_drc_plot_df = []
final_drc_parameters = []
for k, d in drcs.items():
    final_drc_plot = d._sample_curve_df
    final_drc_plot["PlateLabel"] = k[0]
    final_drc_plot["Replicates"] = k[1]
    final_drc_plot_df.append(final_drc_plot)

    params = d.parameters
    params["Log10Inflection"] = params["Inflection"]
    params["Inflection"] = 10 ** params["Inflection"]
    params["LD50"] = np.exp(d.ld_quantile(1-0.5))
    params["LD5"] = np.exp(d.ld_quantile(1-0.05))
    params["LD95"] = np.exp(d.ld_quantile(1-0.95))
    params.update({"PlateLabel": k[0], "Replicates": k[1]})
    final_drc_parameters.append(params)

final_drc_parameters = pd.DataFrame(final_drc_parameters)
final_drc_plot_df = pd.concat(final_drc_plot_df)
final_drc_plot_df = final_drc_plot_df[final_drc_plot_df["x"] <= ann_df["LogDose"].max()]
final_drc_plot_df = final_drc_plot_df[final_drc_plot_df["x"] >= ann_df["LogDose"].min()]

final_drc_plot = (
    alt.Chart(final_drc_plot_df)
    .mark_line()
    .encode(
        alt.X("x:Q", sort="ascending", title="Log Dose"),
        alt.Y("y"),
        color="Replicates",
        strokeDash="PlateLabel",
    )
)

# st.altair_chart(plot_chart, use_container_width=True)
st.altair_chart(plot_chart + final_drc_plot, use_container_width=True)
st.dataframe(final_drc_parameters)

st.download_button(
   "Press to Download (CSV, can open in Excel)",
   final_drc_parameters.to_csv(index=False).encode('utf-8'),
   "file.csv",
   "text/csv",
   key='download-csv'
)

st.subheader("Grouped DRC plots")


split_options_dict = {
    "Replicate Groups": "Replicates",
    "Plate Labels": "PlateLabel",
}
splitting_options = st.multiselect(
    "How to split plots:",
    list(split_options_dict.keys())[:2],
    list(split_options_dict.keys()),
)

if len(splitting_options) == 0:
    st.warning("Please select at least one option")
else:
    splitting_options = [split_options_dict[x] for x in splitting_options]
    split_df = final_drc_plot_df.groupby(splitting_options)
    split_ann = ann_df.groupby(splitting_options)
    for (i, k), (ii, kk) in zip(split_df, split_ann):
        if len(splitting_options) == 1:
            label = f"{splitting_options[0]}: {i}"
        else:
            label = " ".join([f"{x}: {y}" for x, y in zip(splitting_options, i)])

        color_var = (
            "PlateLabel" if "PlateLabel" not in splitting_options else "Replicates"
        )
        drc_plot = (
            alt.Chart(k)
            .mark_line()
            .encode(
                alt.X("x:Q", sort="ascending", title="Log Dose"),
                alt.Y("y"),
                color=color_var,
                strokeDash="PlateLabel",
            )
            .properties(title=label)
        )
        ann_plot = (
            alt.Chart(kk)
            .mark_circle(size=20)
            .encode(
                alt.X("LogDose:Q", sort="ascending", title="Log Dose"),
                alt.Y("Intensity"),
                color=color_var,
            )
        )
        st.altair_chart(drc_plot + ann_plot, use_container_width=True)
