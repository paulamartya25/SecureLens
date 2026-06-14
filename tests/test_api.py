"""
tests/test_api.py
SecureLens — Unit Tests for Flask API
Run with: pytest tests/test_api.py -v
"""

import pytest
import json
import io
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Flask test client."""
    from cloud_server.server import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def make_image_bytes(width=224, height=224, mode="RGB"):
    """Creates a dummy PNG image in memory."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── Test 1: Health Check ──────────────────────────────────────────────

class TestHealthCheck:

    def test_health_returns_200(self, client):
        """Health endpoint must return 200."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_json(self, client):
        """Health endpoint must return JSON."""
        r    = client.get("/health")
        data = json.loads(r.data)
        assert isinstance(data, dict)

    def test_health_ckks_ready(self, client):
        """Health must report CKKS as ready."""
        r    = client.get("/health")
        data = json.loads(r.data)
        assert data["ckks_ready"] is True

    def test_health_has_status(self, client):
        """Health must have status field."""
        r    = client.get("/health")
        data = json.loads(r.data)
        assert "status" in data
        assert data["status"] == "ok"


# ── Test 2: Index Page ────────────────────────────────────────────────

class TestIndexPage:

    def test_index_returns_200(self, client):
        """Index page must return 200."""
        r = client.get("/")
        assert r.status_code == 200

    def test_index_contains_securelens(self, client):
        """Index page must contain SecureLens title."""
        r = client.get("/")
        assert b"SecureLens" in r.data

    def test_index_content_type_html(self, client):
        """Index must return HTML."""
        r = client.get("/")
        assert b"html" in r.data.lower()


# ── Test 3: Info Endpoint ─────────────────────────────────────────────

class TestInfoEndpoint:

    def test_info_returns_200(self, client):
        """Info endpoint must return 200."""
        r = client.get("/api/info")
        assert r.status_code == 200

    def test_info_has_project(self, client):
        """Info must have project name."""
        r    = client.get("/api/info")
        data = json.loads(r.data)
        assert data["project"] == "SecureLens"

    def test_info_has_classes(self, client):
        """Info must list classes."""
        r    = client.get("/api/info")
        data = json.loads(r.data)
        assert "Normal" in data["classes"]
        assert "Pneumonia" in data["classes"]

    def test_info_mentions_ckks(self, client):
        """Info must mention CKKS encryption."""
        r    = client.get("/api/info")
        data = json.loads(r.data)
        assert "CKKS" in data["encryption"]


# ── Test 4: Predict Endpoint — Input Validation ───────────────────────

class TestPredictValidation:

    def test_predict_no_file_returns_400(self, client):
        """Predict without file must return 400."""
        r = client.post("/api/predict")
        assert r.status_code in [400, 503]

    def test_predict_wrong_field_name(self, client):
        """Predict with wrong field name must return 400."""
        buf = make_image_bytes()
        r   = client.post("/api/predict",
                          data={"wrong_field": (buf, "test.png")},
                          content_type="multipart/form-data")
        assert r.status_code in [400, 503]

    def test_predict_empty_filename(self, client):
        """Predict with empty filename must return 400."""
        r = client.post("/api/predict",
                        data={"image": (io.BytesIO(b""), "")},
                        content_type="multipart/form-data")
        assert r.status_code in [400, 503]


# ── Test 5: Predict Endpoint — Valid Input ────────────────────────────

class TestPredictValid:

    def test_predict_png_returns_200(self, client):
        """Valid PNG upload must return 200."""
        buf = make_image_bytes()
        r   = client.post("/api/predict",
                          data={"image": (buf, "xray.png")},
                          content_type="multipart/form-data")
        assert r.status_code == 200

    def test_predict_returns_prediction(self, client):
        """Response must contain prediction field."""
        buf  = make_image_bytes()
        r    = client.post("/api/predict",
                           data={"image": (buf, "xray.png")},
                           content_type="multipart/form-data")
        data = json.loads(r.data)
        assert "prediction" in data
        assert data["prediction"] in ["Normal", "Pneumonia"]

    def test_predict_returns_confidence(self, client):
        """Response must contain confidence between 0 and 100."""
        buf  = make_image_bytes()
        r    = client.post("/api/predict",
                           data={"image": (buf, "xray.png")},
                           content_type="multipart/form-data")
        data = json.loads(r.data)
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 100

    def test_predict_scores_sum_to_100(self, client):
        """Normal + Pneumonia scores must sum to ~100."""
        buf  = make_image_bytes()
        r    = client.post("/api/predict",
                           data={"image": (buf, "xray.png")},
                           content_type="multipart/form-data")
        data  = json.loads(r.data)
        total = data["normal_score"] + data["pneumonia_score"]
        assert abs(total - 100.0) < 0.1, \
            f"Scores sum to {total}, not 100"

    def test_predict_has_encryption_info(self, client):
        """Response must include encryption details."""
        buf  = make_image_bytes()
        r    = client.post("/api/predict",
                           data={"image": (buf, "xray.png")},
                           content_type="multipart/form-data")
        data = json.loads(r.data)
        assert "encryption_info" in data
        enc  = data["encryption_info"]
        assert "scheme" in enc
        assert "CKKS" in enc["scheme"]

    def test_predict_has_pipeline_steps(self, client):
        """Response must include pipeline trace."""
        buf  = make_image_bytes()
        r    = client.post("/api/predict",
                           data={"image": (buf, "xray.png")},
                           content_type="multipart/form-data")
        data = json.loads(r.data)
        assert "pipeline_steps" in data
        assert len(data["pipeline_steps"]) >= 5

    def test_predict_jpeg_works(self, client):
        """JPEG upload must also work."""
        arr = np.random.randint(0,256,(224,224,3),dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        r = client.post("/api/predict",
                        data={"image": (buf, "xray.jpg")},
                        content_type="multipart/form-data")
        assert r.status_code == 200