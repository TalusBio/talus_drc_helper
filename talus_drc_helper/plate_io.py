from io import StringIO

import pandas as pd


def read_synergy_str(lines: list[str]) -> pd.DataFrame:
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


def read_synergy_txt(filepath) -> pd.DataFrame:
    """Reads the text file from the Synergy to a dataframe."""
    with open(file=filepath) as f:
        lines = f.read()

    lines = lines.split("\n")
    return read_synergy_str(lines)
