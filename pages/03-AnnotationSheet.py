from __future__ import annotations

import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from pandas import ExcelFile

from talus_drc_helper.fit import DRCEstimator
from talus_drc_helper.gr_tools import calc_gr
from talus_drc_helper.plate_annotations import (
    AnnotatedPlate,
)

REQUIRED_METADATA = {"plate_id", "user", "protocol_id"}


def get_data() -> tuple[AnnotatedPlate, dict[str, int]]:
    st.markdown("## Input Data\n")
    infile = st.file_uploader("Input file to use", accept_multiple_files=False)
    if infile is None:
        st.warning("No files uploaded")
        st.stop()

    with BytesIO(infile.getvalue()) as bio:
        excel_file = ExcelFile(bio)
        tmp = pd.read_excel(excel_file, sheet_name="cell_count__time0")
        count_t0_map = dict(zip(tmp["cell_line"], tmp["cell_count__time0"]))

        metadata = pd.read_excel(excel_file, sheet_name="metadata", header=0)
        metadata.columns = [x.lower() for x in metadata.columns]
        metadata_dict = dict(zip(metadata["key"], metadata["value"]))
        metadata_dict = {
            k.lower().replace(" ", "_"): v for k, v in metadata_dict.items()
        }

    if not REQUIRED_METADATA.issubset(set(metadata_dict.keys())):
        not_found = REQUIRED_METADATA - set(metadata_dict.keys())
        st.error(
            "Missing required metadata, please make sure you have the following keys:"
            f" {REQUIRED_METADATA}"
        )
        st.error(f"Missing keys: {not_found}")
        st.info("Metadata:")
        st.dataframe(metadata)
        st.stop()
    ann = AnnotatedPlate.from_excel_bytes(infile.getvalue(), skip_bad=True)
    return ann, count_t0_map, metadata


with st.sidebar:
    st.title("Data!")
    ann, count_t0_map, metadata = get_data()
    print(metadata)

tmp = ann.as_df(drop_missing_cols=["compound"])["compound"].unique()

if "DMSO" in tmp:
    normalization_regex = "DMSO"
elif "CTRL" in tmp:
    normalization_regex = "CTRL"
else:
    raise ValueError("Normalization has to be either to 'CTRL' or 'DMSO'")

tmp = ann.widen_norm_column(
    grouping_cols=["cell_line"],
    normalization_column="compound",
    value_column="cell_count",
    normalization_regex=normalization_regex,
    keep_control=True,
)
tmp.rename(columns={"NORM_FACTOR": "cell_count__CTRL"}, inplace=True)
tmp["cell_count__time0"] = tmp["cell_line"].map(count_t0_map)
tmp["viability_percent"] = 100 * tmp["cell_count"] / tmp["cell_count__CTRL"]
tmp["gr_value"] = calc_gr(
    x=tmp["cell_count"], x0=tmp["cell_count__time0"], xctrl=tmp["cell_count__CTRL"]
)


gr_drc = {}
count_drc = {}

gr_df = []
count_df = []

cell_line_grouped_plots = {}
compound_grouped_plots = {}

with tempfile.TemporaryDirectory() as ASDASDASD:
    Path(ASDASDASD).mkdir(parents=True, exist_ok=True)
    for i, sdf in tmp.groupby(["cell_line", "compound"]):
        if np.all(sdf["concentration_in_um"].isna()):
            continue

        if np.all(sdf["cell_count__time0"].isna()):
            continue

        gr_est = DRCEstimator(log_transform_x=True)
        viab_est = DRCEstimator(log_transform_x=True)
        gr_est.fit(X=sdf["concentration_in_um"].values, y=sdf["gr_value"].values)
        viab_est.fit(
            X=sdf["concentration_in_um"].values, y=sdf["viability_percent"].values / 100
        )

        gr_drc[tuple(i)] = gr_est
        count_drc[tuple(i)] = viab_est

        gr_sdf = gr_est.report_df().rename(
            columns={
                "Log10Inflection": "Log10GR50_uM",
                "AbsIC_95": "GR95",
                "AbsIC_50": "GR50",
                "Bias": "GRInf",
                "Inflection": "GEC50",
                "NormalizedResiduals": "NormalizedResidualsGR",
            }
        )[["Log10GR50_uM", "GR95", "GR50", "GEC50", "GRInf", "NormalizedResidualsGR"]]
        gr_sdf["cell_line"] = i[0]
        gr_sdf["compound"] = i[1]

        count_sdf = viab_est.report_df().rename(
            columns={
                "Log10Inflection": "Log10IC50_uM",
                "AbsIC_95": "IC95",
                "AbsIC_50": "IC50",
                "Bias": "ICInf",
                "Inflection": "EC50",
                "NormalizedResiduals": "NormalizedResidualsViability",
            }
        )[
            [
                "Log10IC50_uM",
                "IC95",
                "IC50",
                "EC50",
                "ICInf",
                "NormalizedResidualsViability",
            ]
        ]
        count_sdf["cell_line"] = i[0]
        count_sdf["compound"] = i[1]

        if i[0] not in cell_line_grouped_plots:
            cell_line_grouped_plots[i[0]] = {
                "gr": plt.subplots(1, 1, figsize=(10, 7)),
                "viab": plt.subplots(1, 1, figsize=(10, 7)),
            }

        if i[1] not in compound_grouped_plots:
            compound_grouped_plots[i[1]] = {
                "gr": plt.subplots(1, 1, figsize=(10, 7)),
                "viab": plt.subplots(1, 1, figsize=(10, 7)),
            }

        plot_title = f"{i[0]} - {i[1]}"
        plot_prefix = f"{i[0]}_{i[1]}"

        axs_use = [
            None,
            compound_grouped_plots[i[1]]["gr"][1],
            cell_line_grouped_plots[i[0]]["gr"][1],
        ]
        for ax in axs_use:
            gr_est._plot(
                title=plot_title,
                target_file=f"{ASDASDASD}/{plot_prefix}_gr.png",
                X=sdf["concentration_in_um"].values,
                y=sdf["gr_value"].values,
                ax=ax,
            )
        axs_use = [
            None,
            compound_grouped_plots[i[1]]["viab"][1],
            cell_line_grouped_plots[i[0]]["viab"][1],
        ]
        for ax in axs_use:
            viab_est._plot(
                title=plot_title,
                target_file=f"{ASDASDASD}/{plot_prefix}_viab.png",
                X=sdf["concentration_in_um"].values,
                y=sdf["viability_percent"].values / 100,
                ax=ax,
            )

        gr_df.append(gr_sdf)
        count_df.append(count_sdf)

    gr_df = pd.concat(gr_df)
    count_df = pd.concat(count_df)

    out_df = tmp.merge(gr_df, on=["cell_line", "compound"], how="left").merge(
        count_df, on=["cell_line", "compound"], how="left"
    )

    st.subheader("Plate")
    with st.expander("View Plate Annotations"):
        st.altair_chart(ann.plot(), use_container_width=True)

    st.header("Data")
    st.dataframe(out_df, use_container_width=True)

    out_df.sort_values(["compound", "cell_line", "concentration_in_um"]).to_csv(
        f"{ASDASDASD}/out.csv", index=False
    )
    for k, v in cell_line_grouped_plots.items():
        v["gr"][1].legend(loc="lower left")
        v["viab"][1].legend(loc="lower left")
        v["gr"][0].savefig(f"{ASDASDASD}/celllinegrouped_{k}_gr.png")
        v["viab"][0].savefig(f"{ASDASDASD}/celllinegrouped_{k}_viab.png")

    for k, v in compound_grouped_plots.items():
        v["gr"][1].legend(loc="lower left")
        v["viab"][1].legend(loc="lower left")
        v["gr"][0].savefig(f"{ASDASDASD}/compoundgrouped_{k}_gr.png")
        v["viab"][0].savefig(f"{ASDASDASD}/compoundgrouped_{k}_viab.png")

    with tempfile.TemporaryDirectory() as out_tmpdir:
        Path(out_tmpdir).mkdir(parents=True, exist_ok=True)
        shutil.make_archive(
            f"{out_tmpdir}/bundle", "zip", root_dir=ASDASDASD, base_dir="."
        )
        with open(f"{out_tmpdir}/bundle.zip", "rb") as f2:
            st.download_button(
                "DOWNLOAD EVERYTHING!!! WOOOO!!!", f2.read(), "bundle.zip"
            )
