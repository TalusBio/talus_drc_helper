from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import BytesIO
from string import ascii_uppercase
from typing import Any, Literal

import altair as alt
import numpy as np
import pandas as pd
from loguru import logger
from pandas import ExcelFile, ExcelWriter

from talus_drc_helper.fit import DRCEstimator, DRCEstimatorGroup

LAYOUT = Literal["Horizontal", "Vertical"]
DIRECTION = Literal["TB", "BT", "LR", "CLR", "LRC"]
PLATE_SIZE = Literal["96_well", "384_well"]
DIMENSIONS_96_WELL = (8, 12)
DIMENSIONS_384_WELL = (16, 24)
PLATE_SIZES: dict[PLATE_SIZE, tuple[int, int]] = {
    "96_well": DIMENSIONS_96_WELL,
    "384_well": DIMENSIONS_384_WELL,
}


@dataclass
class PlateAnnotation:
    """A class for annotating plates.

    Parameters
    ----------
    plate_df : pd.DataFrame
        A dataframe with the annotation. The index should be the row
        names, and the columns should be the column names.
    annotation_name : str
        The name of the annotation.

    Examples
    --------
    >>> foo = PlateAnnotation.sample_96()
    >>> # foo.plate_df.head()
    #          1         2         3         4   ...
    # A  0.850061  0.079929  0.280582  0.222699  ...
    # B  0.554867  0.734452  0.928366  0.542993  ...
    # C  0.132042  0.877312  0.620461  0.702565  ...
    # D  0.434650  0.808087  0.462098  0.179796  ...
    # E  0.182519  0.458260  0.912046  0.429932  ...
    >>> foo.annotation_name
    'Sample Annotation'
    >>> # foo.as_long_df().head()
    #       Row  Col  Sample Annotation
    # 0   A    1           0.161664
    # 1   B    1           0.339452
    # 2   C    1           0.902536
    # 3   D    1           0.022280
    # 4   E    1           0.842034
    """

    plate_df: pd.DataFrame
    annotation_name: str

    def __post_init__(self):
        col_range = {str(x) for x in range(1, 26)}
        index_range = set(ascii_uppercase)
        self.plate_df.index = self.plate_df.index.str.upper()
        self.plate_df.columns = self.plate_df.columns.astype(str)
        self.plate_df.columns = self.plate_df.columns.str.upper()
        for x in self.plate_df.index:
            if x not in index_range:
                raise ValueError(f"'{x}' not in {index_range}, unexpected row name")

        for c in self.plate_df.columns:
            if c not in col_range:
                raise ValueError(f"'{c}' not in {col_range}, unexpected column name")

    def as_long_df(self) -> pd.DataFrame:
        """Converts the annotation to a long dataframe.

        Examples
        --------
        >>> foo = PlateAnnotation.sample_96()
        >>> type(foo)
        <class 'talus_drc_helper.plate_annotations.PlateAnnotation'>
        >>> foo.shape
        (8, 12)
        >>> foo.as_long_df().shape
        (96, 3)
        """
        out = self.plate_df.reset_index(names="Row").melt(
            id_vars="Row", var_name="Col", value_name=self.annotation_name
        )
        out = out.astype({"Col": "int"}).sort_values(["Col", "Row"])
        # out['Col'] = [str(x) for x in out['Col']]
        return out

    def to_excel(self, excel_writter: os.PathLike | ExcelWriter) -> None:
        """Saves the annotation to an excel file.

        Parameters
        ----------
        excel_writter : str
            The path to save the annotation to.

        Examples
        --------
        >>> foo = PlateAnnotation.sample_96()
        >>> foo.to_excel("foo.xlsx")
        """
        self.plate_df.to_excel(excel_writter, sheet_name=self.annotation_name)

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of the plate being annotated."""
        return self.plate_df.shape

    def plot(self) -> alt.Chart:
        """Plots the plate as an altair chart.

        Examples
        --------
        >>> foo = PlateAnnotation.sample_96()
        >>> chart = foo.plot()
        >>> type(chart)
        <class 'altair.vegalite.v4.api.Chart'>
        """
        chart_data = self.as_long_df()
        chart = (
            alt.Chart(chart_data)
            .mark_rect()
            .encode(x="Col:O", y="Row:O", color=self.annotation_name)
            .properties(title=self.annotation_name)
        )
        return chart

    @classmethod
    def sample_384(cls, annot_name: str = "Sample Annotation") -> PlateAnnotation:
        """Returns a random 384 well plate annotation."""
        df = pd.DataFrame(
            np.random.rand(*DIMENSIONS_384_WELL),
            index=[ascii_uppercase[i] for i in range(16)],
            columns=[x + 1 for x in range(24)],
        )
        return PlateAnnotation(df, annot_name)

    @classmethod
    def sample_96(cls, annot_name: str = "Sample Annotation") -> PlateAnnotation:
        """Returns a random 96 well plate annotation."""
        df = pd.DataFrame(
            np.random.rand(*DIMENSIONS_96_WELL),
            index=[ascii_uppercase[i] for i in range(8)],
            columns=[x + 1 for x in range(12)],
        )
        return PlateAnnotation(df, annot_name)

    @classmethod
    def from_long_df(cls, df: pd.DataFrame, annotation_col: str) -> PlateAnnotation:
        """Creates a PlateAnnotation from a long dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            A long dataframe with the annotation. The index should be the row
            names, and the columns should be the column names.
        annotation_col : str
            The name of the annotation column.

        Examples
        --------
        >>> foo = PlateAnnotation.sample_96()
        >>> # foo.as_long_df().head()
        # Row  Col  Sample Annotation
        # 0   A    1           0.719382
        # 1   B    1           0.792415
        # 2   C    1           0.682716
        # 3   D    1           0.207254
        # 4   E    1           0.051039
        >>> foo2 = PlateAnnotation.from_long_df(foo.as_long_df(), 'Sample Annotation')
        >>> # foo2.plate_df.head()
        # Col   1     2     3     4     ...
        # Row
        # A    0.0   8.0  16.0  24.1    ...
        # B    1.0   9.0  17.0  25.0    ...
        # C    2.0  10.0  18.0  26.0    ...
        # D    3.0  11.0  19.0  27.0    ...
        # E    4.0  12.0  20.0  28.0    ...
        """
        vals = (
            df[[annotation_col, "Col", "Row"]]
            .reset_index(drop=True)
            .copy()
            .pivot(index="Row", columns="Col")
        )
        tmp_df = pd.DataFrame(
            vals.values,
            index=vals.index,
            columns=vals.columns.droplevel(0),
            copy=True,
        )
        tmp = cls(tmp_df, annotation_name=annotation_col)
        return tmp

    @classmethod
    def join(cls, *annotations: PlateAnnotation) -> pd.DataFrame:
        """Combines annotations and returns a dataframe.

        Parameters
        ----------
        annotations: PlateAnnotation
            The annotations you want to join.

        Examples
        --------
        >>> foo1 = PlateAnnotation.sample_96()
        >>> foo2 = PlateAnnotation.sample_96()
        >>> foo2.annotation_name = 'Other Annotation'
        >>> foo3 = PlateAnnotation.join(foo1, foo2)
        >>> foo3.columns
        Index(['Row', 'Col', 'Sample Annotation', 'Other Annotation'], dtype='object')
        >>> foo3.shape
        (96, 4)
        >>> type(foo3)
        <class 'pandas.core.frame.DataFrame'>
        """
        out = annotations[0].as_long_df().set_index(["Row", "Col"])
        for ann in annotations[1:]:
            out = out.join(ann.as_long_df().set_index(["Row", "Col"]), how="left")

        return out.reset_index()

    @classmethod
    def from_dilution(  # noqa: C901
        cls,
        dil_factor: float,
        initial_dose: float,
        zero_position: Literal["first", "last"],
        direction: DIRECTION,
        skip_edges: bool = True,
        plate_size: PLATE_SIZE = "384_well",
    ) -> PlateAnnotation:
        """Generates a plate annotation from a theoretical dilution.

        Examples
        --------
        >>> out = PlateAnnotation.from_dilution(
        ...     2,
        ...     100,
        ...     'first',
        ...     direction="CLR",
        ...     skip_edges=True,
        ...     plate_size='96_well')
        >>> out.shape
        (8, 10)
        """
        # TODO refactor this ... this is probably the worse python I have written
        # in a while.
        if direction in ["TB", "BT"]:
            num_dilutions = PLATE_SIZES[plate_size][0]
            inner_names = [ascii_uppercase[i] for i in range(num_dilutions)]
            outer_names = [i + 1 for i in range(PLATE_SIZES[plate_size][1])]

            if skip_edges:
                num_dilutions -= 2
                inner_names = inner_names[1:-1]
                outer_names = outer_names[1:-1]

        elif direction in ["CLR", "LRC", "LR"]:
            num_dilutions = PLATE_SIZES[plate_size][1]
            inner_names = [i + 1 for i in range(num_dilutions)]
            outer_names = [
                ascii_uppercase[i] for i in range(PLATE_SIZES[plate_size][0])
            ]
            if direction in ["LR"]:
                if skip_edges:
                    num_dilutions -= 2
                    inner_names = inner_names[1:-1]

            elif direction in ["CLR", "LRC"]:
                num_dilutions //= 2

                if skip_edges:
                    num_dilutions -= 1
                    inner_names = inner_names[1:-1]
        else:
            raise NotImplementedError

        doses = [initial_dose / (dil_factor**x) for x in range(num_dilutions)]
        if zero_position == "first":
            doses = [0] + doses[:-1]
        elif zero_position == "last":
            doses = doses[:-1] + [0]
        else:
            raise NotImplementedError

        if direction == "BT":
            doses = doses[::-1]
        elif direction == "CLR":
            doses = doses[::-1] + doses
        elif direction in ["CLR", "LRC"]:
            doses = doses + doses[::-1]

        assert len(doses) == len(inner_names)
        arr = np.stack([np.array(doses)] * len(outer_names), axis=0)
        cols = inner_names
        ind = outer_names
        if "T" not in direction:
            ind, cols = cols, ind
            arr = arr.T

        out = pd.DataFrame(arr, columns=cols, index=ind)
        if "C" not in out.index:
            out = out.T

        return PlateAnnotation(out, annotation_name="Dose")

    @classmethod
    def from_replicates(
        cls,
        num_replicates: int,
        labels: Iterable,
        layout: LAYOUT,
        skip_edges: bool = True,
        plate_size: PLATE_SIZE = "384_well",
    ) -> PlateAnnotation:
        """Generates a plate annotation from theoretical replicates."""
        plate_dims = PLATE_SIZES[plate_size]
        arr = np.zeros(plate_dims, dtype="object")
        arr[:] = None
        if layout == "Vertical":
            arr = arr.T

        index_offset = 0 if not skip_edges else 1
        for i, lab in enumerate(labels):
            for ii in range(num_replicates):
                index = i * num_replicates
                index += ii
                index += index_offset
                if index < arr.shape[0]:
                    arr[index, :] = lab

        if layout == "Vertical":
            arr = arr.T

        df = pd.DataFrame(
            arr,
            index=[ascii_uppercase[i] for i in range(plate_dims[0])],
            columns=[i + 1 for i in range(plate_dims[1])],
        )
        out = PlateAnnotation(plate_df=df, annotation_name="Replicates")
        return out


@dataclass
class AnnotatedPlate:
    """Combines multiple annotations on a single plate.

    Parameters
    ----------
    annotation : list[PlateAnnotation]
        A list of annotations to combine. These annotations are
        meant to add information at the cell level of the plate.
    plate_level_annotations : dict[str, Any], optional
        A dictionary of plate level annotations, by default {}
        These annotations add information that aplies to the whole
        plate.
    """

    annotation: list[PlateAnnotation]
    plate_level_annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Checks that the annotations are compatible."""
        if len(self.annotation) == 0:
            raise ValueError("Must provide at least one annotation.")
        for x in self.annotation:
            if not isinstance(x, PlateAnnotation):
                raise TypeError("All annotations must be PlateAnnotation objects.")

    @classmethod
    def sample_96(cls) -> AnnotatedPlate:
        """Generates a sample 96 well plate."""
        samp1 = PlateAnnotation.sample_96(annot_name="Drug")
        samp2 = PlateAnnotation.sample_96(annot_name="Concentration")
        samp3 = PlateAnnotation.sample_96(annot_name="Intensity")
        samp4 = PlateAnnotation.sample_96(annot_name="Cells")
        samp2.plate_df = (samp2.plate_df * 100).astype(int)
        samp4.plate_df = (samp3.plate_df < 0.5).astype(str)
        samp3.plate_df = (samp3.plate_df * 10000).astype(int)
        samp1.plate_df = (samp1.plate_df < 0.1).astype(str)

        return cls([samp1, samp2, samp3, samp4])

    def to_excel(self, excel_writter: os.PathLike | ExcelWriter) -> None:
        """Writes the annotations to an excel file."""
        if isinstance(excel_writter, ExcelWriter):
            writer = excel_writter
        else:
            writer = ExcelWriter(excel_writter)

        for x in self.annotation:
            x.plate_df.to_excel(excel_writer=writer, sheet_name=x.annotation_name)

        table_metadata = pd.DataFrame(
            self.plate_level_annotations.items(), columns=["Key", "Value"]
        )
        table_metadata.to_excel(writer, sheet_name="metadata", index=False)
        writer.close()

    def to_excel_bytes(self) -> bytes:
        """Returns the excel file as bytes."""
        with BytesIO() as bio:
            self.to_excel(bio)
            out = bio.getvalue()
        return out

    @staticmethod
    def from_excel_bytes(excel_bytes: bytes, skip_bad: bool = False) -> AnnotatedPlate:
        """Reads the excel file from bytes."""
        with BytesIO(excel_bytes) as bio:
            out = AnnotatedPlate.from_excel(bio, skip_bad=skip_bad)
        return out

    @staticmethod
    def from_excel(
        excel_reader: os.PathLike | ExcelFile, skip_bad: bool = False
    ) -> AnnotatedPlate:
        """Reads the annotations from an excel file."""
        if isinstance(excel_reader, ExcelFile):
            reader = excel_reader
        else:
            reader = ExcelFile(excel_reader)

        out = []
        failed_import = []
        for x in reader.sheet_names:
            try:
                df = pd.read_excel(reader, sheet_name=x, index_col=0)
                out.append(PlateAnnotation(df, annotation_name=x))
            except ValueError as e:
                if "unexpected" in str(e) and "name" in str(e) and skip_bad:
                    failed_import.append(x)
                else:
                    raise e

        if failed_import:
            logger.warning("Failed to import sheets: {}", failed_import)
        return AnnotatedPlate(out)

    def fit_drcs(self) -> None:
        """Fits dose response curves to all annotations."""
        df = self.as_df()
        df = df.dropna()
        [x for x in df.columns if x not in ["Row", "Col", "Dose"]]
        raise NotImplementedError

    def as_df(self, drop_missing_cols: list[str] | None = None) -> pd.DataFrame:
        """Returns a dataframe with all annotations.

        Parameters
        ----------
        drop_missing_cols : list[str] | None, optional
            A list of columns to drop if they are missing, by default None

        Examples
        --------
        >>> out = AnnotatedPlate.sample_96().as_df()
        >>> out.shape
        (96, 6)
        >>> out.columns
        Index(['Row', 'Col', 'Drug', 'Concentration', 'Intensity', 'Cells'], dtype='object')
        >>> # out.head()
        # Row  Col   Drug  Concentration  Intensity   Cells
        # 0   A    1  False       0.399422       3703 False
        # 1   B    1  False       0.783672       7444 False
        # 2   C    1  False       0.080015        788 False
        # 3   D    1  False       0.084008       7729 True
        # 4   E    1  False       0.467157       3771 True
        >>> out = AnnotatedPlate.sample_96()
        >>> out.plate_level_annotations["PlateName"] = "SamplePlate"
        >>> out.plate_level_annotations["PlateID"] = 1
        >>> # out.as_df().head()
        # Row  Col   Drug  Concentration  Intensity   Cells PlateName  PlateID
        # 0   A    1  False       0.399422       3703 False  SamplePlate        1
        # 1   B    1  False       0.783672       7444 False  SamplePlate        1
        # ....
        """
        df = PlateAnnotation.join(*self.annotation)
        for k, v in self.plate_level_annotations.items():
            df[k] = v

        if drop_missing_cols is not None:
            df = df.dropna(subset=drop_missing_cols)

        return df

    def plot(self) -> alt.Chart:
        subcharts = [x.plot() for x in self.annotation]
        return alt.vconcat(*subcharts)

    def append(self, other: PlateAnnotation) -> None:
        """Appends another annotation to the plate."""
        self.annotation.append(other)

    def extend(self, other: list[PlateAnnotation]) -> None:
        """Extends the plate with a list of annotations."""
        self.annotation.extend(other)

    @classmethod
    def from_long_df(cls, df: pd.DataFrame) -> AnnotatedPlate:
        """Creates an annotated plate from a long dataframe."""
        other_cols = [x for x in df.columns if x not in ["Row", "Col"]]
        annots = []
        for col in other_cols:
            annots.append(PlateAnnotation.from_long_df(df, col))
        return cls(annots)

    def widen_norm_column(
        self,
        grouping_cols: list[str] | None,
        normalization_regex: str = "DMSO",
        normalization_column: str = "Compound",
        value_column: str = "Intensity",
        keep_control: bool = False,
    ):
        df = self.as_df(drop_missing_cols=[value_column, normalization_column])

        missing_filter = df[normalization_column].isna()
        if missing_filter.any():
            warnings.warn(
                "Missing values found in value column."
                f" Dropping. {missing_filter.sum()}"
            )

        df = df[~missing_filter].copy().reset_index(drop=True)
        try:
            normalization_filter = df[normalization_column].str.match(
                normalization_regex
            )
        except AttributeError:
            normalization_filter = df[normalization_column] == 0

        normalization_df = df[normalization_filter].copy().reset_index(drop=True)

        if not keep_control:
            to_normalize_df = df[~normalization_filter].copy().reset_index(drop=True)
        else:
            to_normalize_df = df.copy().reset_index(drop=True)

        if len(normalization_df) == 0:
            raise ValueError("No normalization compound found.")

        if len(to_normalize_df) == 0:
            raise ValueError("No compounds to normalize.")

        if isinstance(grouping_cols, str):
            grouping_cols = [grouping_cols]

        if grouping_cols is not None and normalization_column in grouping_cols:
            warnings.warn("Normalization column is in grouping columns. Will remove it")
            grouping_cols = [x for x in grouping_cols if x != normalization_column]

        if grouping_cols is None or len(grouping_cols) == 0:
            norm_factor = normalization_df[value_column].mean()
            to_normalize_df["NORM_FACTOR"] = norm_factor

        else:
            norm_factor_df = normalization_df[grouping_cols + [value_column]]
            norm_factor_df.dropna(axis=1, how="all", inplace=True)
            norm_grouping_cols = [
                x for x in grouping_cols if x in norm_factor_df.columns
            ]

            norm_factor = (
                norm_factor_df.groupby(norm_grouping_cols)
                .mean(numeric_only=False)[value_column]
                .reset_index()
            )

            to_normalize_df = to_normalize_df.merge(norm_factor, on=norm_grouping_cols)
            to_normalize_df.rename(
                columns={
                    value_column + "_y": "NORM_FACTOR",
                    value_column + "_x": value_column,
                },
                inplace=True,
            )

        return to_normalize_df

    def normalize_to_compound(
        self,
        grouping_cols: list[str] | None,
        normalization_regex: str = "DMSO",
        normalization_column: str = "Compound",
        value_column: str = "Intensity",
        rename_column: str | None = "Viability",
    ) -> AnnotatedPlate:
        """Normalizes all plates to a given compound.

        Parameters
        ----------
        grouping_cols : list[str] | None
            The columns to use to group by. Needs to be specified to prevent bugs.
        normalization_regex : str, optional
            The regex to use to identify the normalization compound, by default "DMSO"
        normalization_column : str, optional
            The column to use to identify the normalization compound, by default "Compound"
        value_column : str, optional
            The column to use to identify the value to normalize, by default "Intensity"
        rename_column : str, optional
            The column to use to identify the normalized value, by default "Viability"

        Returns
        -------
        AnnotatedPlate
            The normalized plate.

        Examples
        --------
        >>> plate = AnnotatedPlate.sample_96()
        >>> # plate.as_df().head()
        # Row  Col   Drug  Concentration  Intensity  Cells
        # 0   A    1  False             97       6834  False
        # 1   B    1  False             79       1097   True
        # 2   C    1  False             14       3684   True
        # 3   D    1  False             70       1073   True
        # 4   E    1   True             71       5984  False
        >>> out = plate.normalize_to_compound(
        ...    grouping_cols=None,
        ...    normalization_regex="True",
        ...    normalization_column="Drug",
        ...    value_column="Intensity",
        ...    rename_column="Viability")
        >>> # out.as_df().head()
        # Row  Col   Drug  Concentration  Intensity  Cells  Viability
        # 0   A    1  False           74.0      386.0   True   0.073644
        # 1   B    1  False           44.0     1274.0   True   0.243064
        # 2   C    1  False           84.0     8345.0  False   1.592123
        # 3   D    1  False           72.0      917.0   True   0.174952
        # 4   E    1  False           71.0     4623.0   True   0.882011
        >>> out2 = plate.normalize_to_compound(
        ...     grouping_cols="Cells",
        ...     normalization_regex="True",
        ...     normalization_column="Drug",
        ...     value_column="Intensity",
        ...     rename_column="Viability")
        >>> # out2.as_df().head()
        # This will look almost the same in this example but will be stratified by
        # cell line!

        """
        to_normalize_df = self.widen_norm_column(
            grouping_cols=grouping_cols,
            normalization_column=normalization_column,
            normalization_regex=normalization_regex,
            value_column=value_column,
        )
        if rename_column is None:
            rename_column = value_column
        to_normalize_df[rename_column] = (
            to_normalize_df[value_column] / to_normalize_df["NORM_FACTOR"]
        )
        del to_normalize_df["NORM_FACTOR"]
        return self.from_long_df(to_normalize_df.reset_index(drop=True).copy())


@dataclass
class AnnotatedPlateSet:
    """A set of annotated plates.

    Parameters
    ----------
    plates : dict[str, AnnotatedPlate]
        The plates to use.
    """

    plates: dict[str, AnnotatedPlate]

    def __post_init__(self):
        """Checks that the plates are all the same size."""
        assert isinstance(self.plates, dict)
        assert len(self.plates) > 0

    def items(self):
        """Returns the plates as a dict."""
        return self.plates.items()

    def values(self):
        """Returns the plates as a dict."""
        return self.plates.values()

    def __getitem__(self, key):
        """Returns the plate with the given key."""
        return self.plates[key]

    def as_df(self, drop_missing_cols: list[str] | None = None) -> pd.DataFrame:
        """Returns the annotated plate set as a dataframe."""
        out = []
        for k, v in self.plates.items():
            tmp = v.as_df(drop_missing_cols=drop_missing_cols)
            tmp["Plate"] = k
            out.append(tmp)
        return pd.concat(out)

    def append(self, plate_annotation: PlateAnnotation, plate_id=None) -> None:
        """Appends a plate annotation to the plate set."""
        for k, v in self.plates.items():
            if plate_id is None or k == plate_id:
                v.append(plate_annotation)

    @classmethod
    def from_dispenser_df(cls, df: pd.DataFrame) -> AnnotatedPlateSet:
        """Creates an annotated plate from a dispenser dataframe."""
        try:
            rows = df["Dispensed well"].str[0]
            cols = df["Dispensed well"].str[1:].astype(int)
            concentration = df["Dispensed concentration"]
            compound = df["Fluid name"]
        except KeyError:
            rows = df["Dispensed_well"].str[0]
            cols = df["Dispensed_well"].str[1:].astype(int)
            concentration = df["Dispensed_concentration"]
            compound = df["Fluid_name"]

        plate_id = df["Plate"]

        unique_plates = plate_id.unique()

        tmp_df = pd.DataFrame(
            {
                "Row": rows,
                "Col": cols,
                "Concentration": concentration,
                "Compound": compound,
                "Plate": plate_id,
            }
        )

        out = {k: {} for k in unique_plates}
        for i, x in tmp_df.groupby("Plate"):
            x = x.set_index(["Row", "Col"])
            x = x.drop("Plate", axis=1)
            for e in ["Concentration", "Compound"]:
                indexing_cols = x[e].to_frame().reset_index()[["Row", "Col"]]
                # Report if there are any duplicates
                if indexing_cols.duplicated().any():
                    warnings.warn("Duplicate values found in df. Check consistency")
                    report = indexing_cols[indexing_cols.duplicated()]
                    warnings.warn(f"Duplicate values: {report}")
                    raise ValueError
                vals = x[e].to_frame().reset_index().pivot(index="Row", columns="Col")
                tmp_df2 = pd.DataFrame(
                    vals.values, index=vals.index, columns=vals.columns.droplevel(0)
                )

                out[i][e] = PlateAnnotation(tmp_df2, annotation_name=e)
        out = {k: AnnotatedPlate(list(v.values())) for k, v in out.items()}
        return AnnotatedPlateSet(out)

    def normalize_to_compound(
        self,
        grouping_cols: list[str] | None,
        normalization_regex: str = "DMSO",
        normalization_column: str = "Compound",
        value_column: str = "Intensity",
        rename_column: str | None = "Viability",
    ) -> AnnotatedPlateSet:
        """Normalizes all plates to a given compound."""
        out = {}
        for i, x in self.plates.items():
            out[i] = x.normalize_to_compound(
                grouping_cols=grouping_cols,
                normalization_regex=normalization_regex,
                normalization_column=normalization_column,
                value_column=value_column,
                rename_column=rename_column,
            )
        return AnnotatedPlateSet(out)

    def fit_drc(
        self,
        target_variable: str,
        grouping_cols: list[str],
        dose_variable: str = "Dose",
        log_transform_x: bool = False,
    ) -> dict[tuple, DRCEstimator]:
        """Fits a DRC to the data."""
        df = self.as_df(
            drop_missing_cols=grouping_cols + [target_variable, dose_variable]
        )

        out = {}
        for i, x in df.groupby(grouping_cols):
            # THis section throws this warning.
            # FutureWarning: In a future version of pandas, a length
            #  1 tuple will be returned when iterating over a groupby
            #  with a grouper equal to a list of length 1. Don't supply
            #  a list with a single grouper to avoid this warning.
            # Nontheless, that is the desidered behavior.
            ys = x[target_variable].values
            X = x[dose_variable].values  # noqa: 806
            estimator = DRCEstimator(log_transform_x=log_transform_x)
            estimator.fit(X, ys)
            if not isinstance(i, tuple):
                i = (i,)
            out[i] = estimator

        out = DRCEstimatorGroup(
            groupings=tuple(out.keys()),
            estimators=tuple(out.values()),
            grouping_variables=tuple(grouping_cols),
            target_variable=target_variable,
            dose_variable=dose_variable,
        )
        return out
