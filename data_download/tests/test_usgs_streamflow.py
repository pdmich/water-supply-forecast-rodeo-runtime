import pandas as pd
import pytest

from wsfr_download import usgs_streamflow


def test_get_daily_values_for_usgs_site_uses_waterdata_and_preserves_shape(monkeypatch, tmp_path):
    response = pd.DataFrame(
        {
            "time": ["2020-10-01", "2020-10-02"],
            "value": [1.2, 2.3],
            "parameter_code": ["00060", "00060"],
            "approval_status": ["Approved", "Provisional"],
            "qualifier": [[], ["ESTIMATED", "ICE"]],
        }
    )
    request = {}

    def fake_get_daily(**kwargs):
        request.update(kwargs)
        return response, None

    monkeypatch.setattr(usgs_streamflow.waterdata, "get_daily", fake_get_daily)

    result = usgs_streamflow.get_daily_values_for_usgs_site(
        usgs_id="01234567",
        forecast_year=2021,
        fy_start_month=10,
        fy_start_day=1,
        fy_end_month=7,
        fy_end_day=22,
    )

    assert request == {
        "monitoring_location_id": "USGS-01234567",
        "parameter_code": "00060",
        "statistic_id": "00003",
        "time": "2020-10-01/2021-07-21",
    }
    assert result.index.name == "datetime"
    assert list(result.columns) == ["site_no", "00060_Mean", "00060_Mean_cd"]
    assert result.index.tolist() == list(pd.to_datetime(response["time"]))
    assert result["site_no"].tolist() == ["01234567", "01234567"]
    assert result["00060_Mean"].tolist() == [1.2, 2.3]
    assert result["00060_Mean_cd"].tolist() == ["Approved", "Provisional,ESTIMATED,ICE"]

    out_file = tmp_path / "streamflow.csv"
    result.to_csv(out_file)
    saved = pd.read_csv(out_file, dtype={"site_no": str}).rename(
        columns={"site_no": "usgs_id", "00060_Mean_cd": "discharge_qa"}
    )
    assert saved[["usgs_id", "00060_Mean", "discharge_qa"]].to_dict("list") == {
        "usgs_id": ["01234567", "01234567"],
        "00060_Mean": [1.2, 2.3],
        "discharge_qa": ["Approved", "Provisional,ESTIMATED,ICE"],
    }


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"approval_status": ["Approved"]}, "Approved"),
        ({"qualifier": [["ESTIMATED"]]}, "ESTIMATED"),
        ({"qualifier": ["ICE"]}, "ICE"),
        ({"approval_status": [None], "qualifier": [None]}, None),
        ({"approval_status": [pd.NA], "qualifier": [[]]}, None),
        ({}, None),
    ],
)
def test_get_daily_values_for_usgs_site_handles_optional_metadata(monkeypatch, metadata, expected):
    response = pd.DataFrame({"time": ["2020-10-01"], "value": [1.2], **metadata})
    monkeypatch.setattr(usgs_streamflow.waterdata, "get_daily", lambda **kwargs: (response, None))

    result = usgs_streamflow.get_daily_values_for_usgs_site("01234567", 2021, 10, 1, 7, 22)

    if expected is None:
        assert pd.isna(result["00060_Mean_cd"].iloc[0])
    else:
        assert result["00060_Mean_cd"].iloc[0] == expected


def test_get_daily_values_for_usgs_site_preserves_empty_output_schema(monkeypatch):
    response = pd.DataFrame(columns=["time", "value", "approval_status", "qualifier"])
    monkeypatch.setattr(usgs_streamflow.waterdata, "get_daily", lambda **kwargs: (response, None))

    result = usgs_streamflow.get_daily_values_for_usgs_site("01234567", 2021, 10, 1, 7, 22)

    assert result.empty
    assert result.index.name == "datetime"
    assert list(result.columns) == ["site_no", "00060_Mean", "00060_Mean_cd"]