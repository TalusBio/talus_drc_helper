from io import StringIO
from pathlib import Path
from string import ascii_uppercase
from typing import Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from talus_drc_helper.plate_annotations import AnnotatedPlate, PlateAnnotation


def read_excel_xml(path):
    """Converts Excel XML to a nested list."""
    # TODO rewrite in lxml ... this is very slow...
    file = open(path).read()
    soup = BeautifulSoup(file, "xml")
    workbook = []
    for sheet in soup.findAll("Worksheet"):
        sheet_as_list = []
        for row in sheet.findAll("Row"):
            row_as_list = []
            for cell in row.findAll("Cell"):
                if cell.Data is not None:
                    cell_text = cell.Data.text
                else:
                    cell_text = ""

                if (cell.Data is not None) and "Index" in cell.attrs:
                    ind = int(cell.attrs["Index"])
                    if len(row_as_list) < (ind):
                        row_as_list += [""] * ((ind) - len(row_as_list))

                    row_as_list[ind - 1] = cell_text
                else:
                    row_as_list.append(cell_text)
            sheet_as_list.append(row_as_list)
        workbook.append(sheet_as_list)
    return workbook


def parse_metadata_sheet(data):
    metadata_dict = {}
    for x in data:
        if len(x) < 2:
            continue
        metadata_dict[x[0]] = ", ".join(x[1:])

    return metadata_dict


def parse_tabular_sheet(data):
    max_width = max(len(x) for x in data)
    tmp_list = [None] * len(data)

    for i, row in enumerate(data):
        tmp_list[i] = row + [""] * (max_width - len(row))

    tmp_df = pd.read_csv(
        StringIO("\n".join([",".join([f'"{y}"' for y in x]) for x in tmp_list])),
        index_col=False,
    )
    tmp_df.columns = [x.replace("\n", "_").replace(" ", "_") for x in tmp_df.columns]
    keep = [
        "Plate_ID",
        "Plate",
        "Dispensed_well",
        "Dispensed_row",
        "Dispensed_col",
        "Fluid_name",
        "Concentration",
        "Total_well_volume_(nL)",
        "Volume_(nL)_DMSO_normalization",
        "DMSO_%",
    ]
    tmp_df = tmp_df[keep]
    undispensed = tmp_df["Total_well_volume_(nL)"].isna()
    tmp_df = tmp_df[~undispensed]
    return tmp_df


def split_on_empty_rows(nested_list):
    """Splits a nested list on empty rows.

    This is meant to be used on a nested list where every element represents a row
    in a table. The function will split the nested list on empty rows, returning
    a list of tables.
    """
    out = []
    current = []
    for row in nested_list:
        if len(row) == 0:
            out.append(current)
            current = []
        else:
            if row:
                current.append(row)

    if current:
        out.append(current)

    return out


def parse_fluids_sheet(nested_list):
    out = split_on_empty_rows(nested_list)
    out = [
        pd.DataFrame(
            x[1:], columns=[y.replace(" ", "_").replace("\n", "_") for y in x[0]]
        )
        for x in out
        if x
    ]
    return out


def read_dispenser_xml(path):
    """Reads and parses the xml output from the dispenser."""
    #
    """
    The xml is an excel with multiple sheets.
    1. Summary
    2. fluids contains 4 tables -> [
        fluid names and concentrations,
        casette loading volumes,
        unit definitions,
        casette types,
    ]
    3. Plates -> [
          'Plate',
          'Plate name',
          'Plate ID',
          'Type',
          'Rows',
          'Columns',
          'Additional volume (µL)',
          'DMSO limit (%)',
          "Don't shake",
          "Don't dispense"]
    4. tabular
    """
    data = read_excel_xml(path)
    metadata = parse_metadata_sheet(data[0])
    tabular = parse_tabular_sheet(data[3])
    fluids_tables = parse_fluids_sheet(data[1])
    plate_metadata = pd.DataFrame(
        data[2][1:],
        columns=[x.replace(" ", "_").replace("\n", "_") for x in data[2][0]],
    )
    detailed_tabular = pd.DataFrame(
        data[4][1:],
        columns=[x.replace(" ", "_").replace("\n", "_") for x in data[4][0]],
    )
    out = {
        "metadata": metadata,
        "tabular": tabular,
        "fluids": fluids_tables,
        "plate_metadata": plate_metadata,
        "detailed_tabular": detailed_tabular,
    }
    return out


def make_templates_from_dispenser_xml(source, target_dir: Optional[Path] = None):
    if target_dir is None:
        target_dir = Path(".")
    out = read_dispenser_xml(source)
    tmp_df = out["tabular"]
    tmp_metadata = out["plate_metadata"]

    metadata_dicts = {
        x[1].to_dict()["Plate"]: x[1].to_dict() for x in tmp_metadata.iterrows()
    }
    for v in metadata_dicts.values():
        v.update(out["metadata"])

    renames = {
        "Concentration": "concentration_in_um",
        "Fluid_name": "compound",
        "DMSO_%": "percentage_dmso",
        "Dispensed_well": "well_label",
    }

    keep = {}
    for plate_num, x in tmp_df.groupby("Plate"):
        x["Dispensed_row"] = np.array(list(ascii_uppercase))[
            x["Dispensed_row"].values - 1
        ]
        # x["Dispensed_col"] = x["Dispensed_col"].astype(str)
        x = x.set_index(["Dispensed_row", "Dispensed_col"])
        x = x.drop("Plate", axis=1).drop("Plate_ID", axis=1)
        tables = {}
        for e in ["Concentration", "Fluid_name", "DMSO_%", "Dispensed_well"]:
            vals = (
                x[e]
                .to_frame()
                .reset_index()
                .pivot(index="Dispensed_col", columns="Dispensed_row")
            )
            tmp_df2 = pd.DataFrame(
                vals.values.T, index=vals.columns.droplevel(0), columns=vals.index
            )
            tmp_df2 = tmp_df2.rename_axis(None, axis=0)

            tables[renames[e]] = PlateAnnotation(tmp_df2, renames[e])
        keep[plate_num] = AnnotatedPlate(
            annotation=list(tables.values()),
            plate_level_annotations=metadata_dicts[str(plate_num)],
        )

    for k, v in keep.items():
        spreadsheet_name = f"{v.plate_level_annotations['Plate']}_{v.plate_level_annotations['Plate_name']}_template.xlsx"
        plots_html_name = f"{v.plate_level_annotations['Plate']}_{v.plate_level_annotations['Plate_name']}_plots.html"
        target_file = target_dir / spreadsheet_name
        target_html = target_dir / plots_html_name
        chart = v.plot()
        chart.save(target_html)
        v.to_excel(str(target_file))
