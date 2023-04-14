from talus_drc_helper.plate_io import read_synergy_txt


def test_reads_synergy_data(shared_datadir):
    """Smoke test to check that the synergy data can be read.

    And that the output has the right number of rows and columns.
    """
    filepath = shared_datadir / "20221122_CTG CAOV3 plate 1.txt"
    out = read_synergy_txt(filepath=filepath)
    assert len(out.columns) == 24
    assert len(out) == 16
