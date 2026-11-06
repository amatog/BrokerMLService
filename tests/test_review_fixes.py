import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from application import application
from train_risk_model import compute_drawdown_and_var, build_temporal_dataset, validate_training_params


def call(path, method="GET", body=None, token=""):
    raw = json.dumps(body or {}).encode()
    status = []
    env = {"PATH_INFO": path, "REQUEST_METHOD": method, "CONTENT_LENGTH": str(len(raw)),
           "wsgi.input": io.BytesIO(raw), "HTTP_AUTHORIZATION": token}
    data = b"".join(application(env, lambda code, headers: status.append(code)))
    return int(status[0].split()[0]), json.loads(data)


class ReviewFixTests(unittest.TestCase):
    def test_risk_contract_and_invalid_positions(self):
        code, result = call("/risk", "POST", {"positions": [
            {"symbol": "AAPL", "quantity": 2, "last_price": 10}]})
        self.assertEqual(code, 200)
        self.assertEqual(result["total_value"], 20)
        self.assertEqual(result["method"], "heuristic")
        self.assertEqual(call("/risk", "POST", {"positions": [
            {"symbol": "AAPL", "quantity": -1, "last_price": 10}]})[0], 400)
        self.assertEqual(call("/trend")[0], 501)

    def test_training_requires_token_and_bounds(self):
        with patch.dict(os.environ, {"ML_ADMIN_TOKEN": "test"}):
            self.assertEqual(call("/train-risk-model", "POST")[0], 403)
            self.assertEqual(call("/train-risk-model", "POST", {"num_portfolios": 1000000}, "Bearer test")[0], 400)
        with self.assertRaises(ValueError):
            validate_training_params(100, 30, 3, ["AAPL"])

    def test_risk_retains_deployed_cash_support(self):
        code, result = call("/risk", "POST", {"cash": 80, "positions": [
            {"symbol": "AAPL", "quantity": 2, "last_price": 10}]})
        self.assertEqual(code, 200)
        self.assertEqual(result["total_value"], 100)
        self.assertAlmostEqual(result["concentration"], 0.2)
        self.assertEqual(call("/risk", "POST", {"cash": -1, "positions": []})[0], 400)

    def test_training_lock_prevents_duplicate_job(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                "ML_ADMIN_TOKEN": "test", "ML_ARTIFACT_DIR": root}):
            Path(root, "training.lock").touch()
            self.assertEqual(call("/train-risk-model", "POST", token="Bearer test")[0], 409)

    def test_first_day_loss_counts(self):
        prices = pd.DataFrame({"AAPL": [100, 79, 79]})
        drawdown, _ = compute_drawdown_and_var({"AAPL": 1.0}, prices, 2)
        self.assertAlmostEqual(drawdown, 0.21)

    def test_temporal_features_exclude_their_label_window(self):
        prices = pd.DataFrame({"AAPL": np.linspace(100, 200, 400)},
                              index=pd.date_range("2020-01-01", periods=400))
        portfolios = [{"AAPL": 1.0}] * 100
        train, test, _, _, meta = build_temporal_dataset(portfolios, prices, 30)
        self.assertLessEqual(meta["train_label_end"], meta["test_feature_asof_start"])
        modified = prices.copy()
        modified.iloc[64:94] *= 0.7
        changed, _, _, _, _ = build_temporal_dataset(portfolios, modified, 30)
        np.testing.assert_array_equal(train[0], changed[0])
        self.assertTrue(len(test))

    def test_training_publishes_model_pair_before_manifest(self):
        from pathlib import Path
        from train_risk_model import train_risk_model
        prices = pd.DataFrame({"AAPL": np.linspace(100, 200, 400)},
                              index=pd.date_range("2020-01-01", periods=400))
        with tempfile.TemporaryDirectory() as root, \
                patch("train_risk_model.load_multiple_symbols", return_value=prices), \
                patch("train_risk_model.simulate_random_portfolios", return_value=[{"AAPL": 1.0}] * 100):
            result = train_risk_model(100, 30, 3, model_path=str(Path(root, "model.pkl")),
                                      feature_path=str(Path(root, "features.pkl")))
            manifest = json.loads(Path(root, "current.json").read_text())
            self.assertTrue(Path(manifest["model"]).is_file())
            self.assertTrue(Path(manifest["features"]).is_file())
            self.assertEqual(manifest["model"], result["model_path"])


if __name__ == "__main__":
    unittest.main()
