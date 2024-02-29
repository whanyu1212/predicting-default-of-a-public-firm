import pandas as pd
import pytest

from src.extract_financial_data import FinancialDataExtractor


@pytest.fixture
def fde():
    return FinancialDataExtractor("AAPL", "2020-01-01", "2020-12-31")


def test_init(fde):
    assert fde.symbol == "AAPL"
    assert fde.start == "2020-01-01"
    assert fde.end == "2020-12-31"
    assert fde.interval == "1d"
    assert isinstance(fde.data, pd.DataFrame)


def test_get_data(fde):
    data = fde.get_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_fill_missing_dates(fde):
    data = fde.get_data()
    filled_data = fde.fill_missing_dates(data)
    assert isinstance(filled_data, pd.DataFrame)
    assert not filled_data.empty


def test_calculate_returns(fde):
    data = fde.get_data()
    filled_data = fde.fill_missing_dates(data)
    returned_data = fde.calculate_returns(filled_data)
    assert isinstance(returned_data, pd.DataFrame)
    assert not returned_data.empty
    assert "Return" in returned_data.columns


def test_extraction_flow(fde):
    data = fde.extraction_flow()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "Return" in data.columns
