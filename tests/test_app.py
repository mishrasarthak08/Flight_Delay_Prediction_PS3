"""Exercise the historical demo and its error feedback without a browser."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "results/reports/demo_flights.csv").exists(),
    reason="Run build_demo.py after training",
)
def test_historical_prediction_and_rescheduling_pages():
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Predict a historical flight").run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 3
    app.sidebar.radio[0].set_value("Rescheduling sensitivity").run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    app.multiselect[0].set_value([30, 60]).run()
    app.button[0].click().run()
    assert not app.exception
    assert "zero-shift" in app.error[0].value


@pytest.mark.skipif(
    not (ROOT / "models/bundle_metadata.json").exists(),
    reason="Corrected model artifacts required",
)
@pytest.mark.parametrize(
    "page", ["Model evaluation", "Training-data EDA", "Method and limitations"]
)
def test_evidence_pages_render_without_exceptions(page):
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
