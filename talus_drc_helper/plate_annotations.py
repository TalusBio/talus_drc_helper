from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd

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
    annotation_name : list[str]
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
        )
        return chart

    @classmethod
    def sample_384(cls) -> PlateAnnotation:
        """Returns a random 384 well plate annotation."""
        df = pd.DataFrame(
            np.random.rand(*DIMENSIONS_384_WELL),
            index=[ascii_uppercase[i] for i in range(16)],
            columns=[x + 1 for x in range(24)],
        )
        return PlateAnnotation(df, "Sample Annotation")

    @classmethod
    def sample_96(cls) -> PlateAnnotation:
        """Returns a random 96 well plate annotation."""
        df = pd.DataFrame(
            np.random.rand(*DIMENSIONS_96_WELL),
            index=[ascii_uppercase[i] for i in range(8)],
            columns=[x + 1 for x in range(12)],
        )
        return PlateAnnotation(df, "Sample Annotation")

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
            out = out.join(ann.as_long_df().set_index(["Row", "Col"]))

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
        arr[:] = ""
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
    """Combines multiple annotations on a single plate."""

    annotation: list[PlateAnnotation]

    def fit_drcs(self) -> None:
        """Fits dose response curves to all annotations."""
        df = self.as_df()
        df = df.dropna()
        [x for x in df.columns if x not in ["Row", "Col", "Dose"]]
        raise NotImplementedError

    def as_df(self) -> pd.DataFrame:
        """Returns a dataframe with all annotations."""
        return PlateAnnotation.join(*self.annotation)
