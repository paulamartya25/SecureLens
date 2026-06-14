"""
utils/audit_log.py
SecureLens — Audit Trail System
Logs all encrypted inferences with timestamps.
"""

import os, json, hashlib, logging
from datetime import datetime

LOGS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE   = os.path.join(LOGS_DIR, "audit_trail.jsonl")
APP_LOG    = os.path.join(LOGS_DIR, "app.log")

# Configure Python logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(APP_LOG),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("securelens")


class AuditLogger:
    """
    Logs every encrypted inference request.
    Stores: timestamp, image hash, prediction,
            confidence, encryption params, latency.
    Never stores: raw image data, patient identifiers.
    """

    def __init__(self, log_file=LOG_FILE):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_inference(
        self,
        image_bytes: bytes,
        prediction: str,
        confidence: float,
        latency_ms: float,
        encryption_params: dict,
        endpoint: str = "/api/predict",
    ):
        """
        Logs one encrypted inference event.
        Image is hashed — never stored raw.
        """
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

        entry = {
            "timestamp"        : datetime.utcnow().isoformat() + "Z",
            "endpoint"         : endpoint,
            "image_hash"       : image_hash,
            "prediction"       : prediction,
            "confidence_pct"   : round(confidence, 2),
            "latency_ms"       : round(latency_ms, 1),
            "encryption_scheme": encryption_params.get(
                "scheme", "CKKS"),
            "security_bits"    : encryption_params.get(
                "security_bits", 128),
            "ciphertext_kb"    : encryption_params.get(
                "ciphertext_size_kb", 0),
            "data_exposed"     : "none",
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(
            f"Inference logged | hash={image_hash} "
            f"pred={prediction} conf={confidence:.1f}% "
            f"latency={latency_ms:.1f}ms")

        return entry

    def get_recent_logs(self, n=20):
        """Returns last n log entries."""
        if not os.path.exists(self.log_file):
            return []
        with open(self.log_file, encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                continue
        return list(reversed(entries))

    def get_stats(self):
        """Returns summary statistics of all logged inferences."""
        if not os.path.exists(self.log_file):
            return {"total": 0}

        entries = []
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    continue

        if not entries:
            return {"total": 0}

        preds     = [e["prediction"] for e in entries]
        latencies = [e["latency_ms"] for e in entries]

        return {
            "total"          : len(entries),
            "normal_count"   : preds.count("Normal"),
            "pneumonia_count": preds.count("Pneumonia"),
            "avg_latency_ms" : round(
                sum(latencies)/len(latencies), 1),
            "max_latency_ms" : round(max(latencies), 1),
            "min_latency_ms" : round(min(latencies), 1),
            "first_log"      : entries[-1]["timestamp"],
            "last_log"       : entries[0]["timestamp"],
        }


# Global instance
audit_logger = AuditLogger()