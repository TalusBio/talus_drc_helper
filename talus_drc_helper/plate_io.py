import os
from io import StringIO

import pandas as pd


def read_synergy_str(lines: list[str]) -> pd.DataFrame:
    """Reads the text file from the Synergy to a dataframe.

    Parameters
    ----------
    lines : list[str]
        The lines of the text file.
    """
    assert any("Synergy Neo2" in x for x in lines)

    metadata = []
    results = []

    results_section = False
    for line in lines:
        if line := line.strip():
            if results_section:
                results.append(line.replace("\tLum", ""))
            else:
                if line == "Results":
                    results_section = True
                metadata.append(line)

    results = pd.read_csv(StringIO("\n".join(results)), sep="\t")
    return results


def read_synergy_txt(filepath: os.PathLike) -> pd.DataFrame:
    """Reads the text file from the Synergy to a dataframe."""
    with open(file=filepath) as f:
        lines = f.read()

    lines = lines.split("\n")
    return read_synergy_str(lines)


def read_dispenser_text(str) -> pd.DataFrame:
    return read_dispenser_excel(StringIO(str))


def read_dispenser_excel(filepath: str) -> pd.DataFrame:
    """Reads the excel file from the dispenser to a dataframe."""
    df = pd.read_excel(filepath, sheet_name="Tabular detail")
    df.columns = [x.replace("\n", " ").strip() for x in df.columns]

    # This section is to deal with the fact that the dispenser
    # will generate dilutions by adding DMSO to the wells. Therefore
    # Multiple compounds are added to each well.
    # This attempts to remove the dmso from the wells that also have a compound.
    df = pd.concat(
        [
            x[x["Dispensed concentration"].isna().__invert__()] if len(x) > 1 else x
            for _, x in df.groupby(["Plate", "Dispensed well"])
        ]
    )
    return df
