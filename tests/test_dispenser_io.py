from pathlib import Path

from talus_drc_helper.dispenser_io import (
    make_templates_from_dispenser_xml,
    read_dispenser_xml,
)


def test_reads_dispenser_data(shared_datadir):
    source = shared_datadir / "dispenser/11032023_4_384_well.xml"
    source = str(source)
    out = read_dispenser_xml(source)


def test_template_generation(shared_datadir, tmpdir):
    source = shared_datadir / "dispenser/11032023_4_384_well.xml"
    source = str(source)
    xl_files = Path(tmpdir).rglob("*.xlsx")
    html_files = Path(tmpdir).rglob("*.html")

    assert len(list(xl_files)) == 0
    assert len(list(html_files)) == 0

    make_templates_from_dispenser_xml(source=source, target_dir=tmpdir)

    xl_files = Path(tmpdir).rglob("*.xlsx")
    html_files = Path(tmpdir).rglob("*.html")

    assert len(list(xl_files)) == 4
    assert len(list(html_files)) == 4
