import altair as alt
import numpy as np
import streamlit as st

from talus_drc_helper.plate_annotations import AnnotatedPlateSet


def normalize_if_wanted(ann_plate_set: AnnotatedPlateSet, grouping_cols):
    st.markdown(
        "## Normalization\nHere you will select how to normalize the data, for instance"
        " if the column 'Drug' has the value 'DMSO' then the data will be normalized to"
        " that value.\n\n**IF** the column selected is the dose, then it will normalize"
        " to values of 0 (and will ignore the regex)."
    )
    if st.checkbox("Normalize data", value=True):
        try:
            ann_plate_set = ann_plate_set.normalize_to_compound(
                grouping_cols=grouping_cols,
                normalization_regex=st.text_input(
                    "Normalization Regex",
                    value="DMSO",
                    help="Regex to match for normalization",
                ),
                normalization_column=st.selectbox(
                    "Normalization Column", options=grouping_cols
                ),
                value_column="Intensity",
                rename_column="Viability",
            )
        except ValueError:
            st.error(
                "Was not able to normalize with those parameters, Please try a"
                " different column or a different value."
            )
            raise
            st.stop()
        if np.any(ann_plate_set.as_df()["Viability"] > 2):
            st.warning(
                "The normalized data has values greater than 2, this is probably not"
                " right"
            )

        return ann_plate_set, "Viability"

    return ann_plate_set, "Intensity"


def fit_and_show_ann_plate(ann_plate: AnnotatedPlateSet):
    ann_plate_df = ann_plate.as_df()

    st.subheader("Complete Data")
    st.dataframe(ann_plate_df, use_container_width=True)

    st.markdown(
        "## Grouping\n"
        "*The grouping means that one dose response should be "
        "calculated for every combination of values in all of the "
        "selected columns. For instance if you select 'Drug' and "
        "'Cell Line' then one dose response will be calculated for "
        "every combination of drug and cell line.* \n"
        "> Right now it will also group by plate"
    )
    group_cols_opts = [
        x
        for x in ann_plate_df.columns
        if x not in ["Row", "Col", "Intensity", "Plate", "Concentration"]
    ]
    group_cols = st.multiselect(
        "Select grouping columns", group_cols_opts, default=group_cols_opts
    )

    for i, (k, v) in enumerate(ann_plate_df.groupby(group_cols)):
        with st.expander(f"View grouping data for group  {i}: {str(k)}"):
            st.dataframe(v)

    ann_plate, y_column = normalize_if_wanted(
        ann_plate, grouping_cols=group_cols + ["Concentration"]
    )
    with st.expander("View normalized data"):
        st.dataframe(ann_plate.as_df())

    st.subheader("DRC")
    dose_var = "Concentration"
    # dose_var = st.selectbox(
    #     "Dose Variable",
    #     options=[x for x in group_cols_opts if x not in group_cols],
    #     index=0,
    # )
    fits = ann_plate.fit_drc(
        target_variable=y_column,
        grouping_cols=group_cols,
        dose_variable=dose_var,
        log_transform_x=True,
    )
    fit_data = ann_plate.as_df(drop_missing_cols=group_cols + [dose_var])
    fit_data = fit_data[fit_data[dose_var] > 0]

    nonmissing_x = fit_data[dose_var].copy().dropna()
    min_x = nonmissing_x.min()
    max_x = nonmissing_x.max()

    curve_data = fits._sample_curve_df
    curve_data = curve_data[curve_data[dose_var] <= max_x]
    curve_data = curve_data[curve_data[dose_var] >= min_x]

    color_var = st.selectbox("Color By", options=group_cols, index=0)
    split_var = st.selectbox(
        "Split By", options=[None] + group_cols, index=len(group_cols)
    )

    def encode_drc(chart):
        out = chart.encode(
            x=alt.X(f"{dose_var}", scale=alt.Scale(type="log")),
            y=alt.Y(f"{y_column}"),
            color=alt.Color(f"{color_var}:N"),
            detail=alt.Detail(f"{color_var}:N"),
        )
        return out

    if split_var is not None:
        for (k1, v1), (k2, v2) in zip(
            curve_data.groupby(split_var), fit_data.groupby(split_var)
        ):
            curve_chart = encode_drc(alt.Chart(v1).mark_line())
            point_chart = encode_drc(alt.Chart(v2).mark_point(size=10))
            st.altair_chart(
                (point_chart + curve_chart).properties(title=f"{split_var}={k1}"),
                use_container_width=True,
            )

    else:
        curve_chart = encode_drc(alt.Chart(curve_data).mark_line())
        point_chart = encode_drc(alt.Chart(fit_data).mark_point(size=10))
        st.altair_chart(
            (point_chart + curve_chart).properties(title="DRCS!!!"),
            use_container_width=True,
        )

    report = fits.report_df()
    st.dataframe(report, use_container_width=True)
    st.download_button(
        "Press to Download (CSV, can open in Excel)",
        report.to_csv(index=False).encode("utf-8"),
        "file.csv",
        "text/csv",
        key="download-csv",
    )
