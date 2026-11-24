import os
import tempfile
import unittest

from app import MAP_EVENTS, app


class AppTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.database_directory = tempfile.TemporaryDirectory()
        app.config["DATABASE"] = os.path.join(self.database_directory.name, "predictions.sqlite3")
        self.client = app.test_client()

    def tearDown(self):
        self.database_directory.cleanup()

    def test_index_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SISMO AI PER", response.data)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_prediction(self):
        response = self.client.post("/api/predict", json={
            "department": "Lima",
            "magnitude": 5.2,
            "depth": 35,
            "distance": 25,
            "k": 5,
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn(data["category"], {"Leve", "Moderada", "Fuerte", "Empate"})
        self.assertEqual(len(data["percentages"]), 3)
        self.assertIsInstance(data["record_id"], int)

    def test_invalid_prediction_parameters(self):
        response = self.client.post("/api/predict", json={
            "magnitude": "NaN",
            "depth": 35,
            "distance": 25,
            "k": 5,
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())
        self.assertEqual(self.client.get("/api/predictions").get_json()["total"], 0)

    def test_each_prediction_is_saved_and_returned_newest_first(self):
        for department in ("Lima", "Cusco"):
            response = self.client.post("/api/predict", json={
                "department": department,
                "magnitude": 5.2,
                "depth": 35,
                "distance": 25,
                "k": 5,
            })
            self.assertEqual(response.status_code, 200)

        history_response = self.client.get("/api/predictions")
        self.assertEqual(history_response.status_code, 200)
        history = history_response.get_json()
        self.assertEqual(history["total"], 2)
        self.assertEqual([item["department"] for item in history["predictions"]], ["Cusco", "Lima"])
        self.assertEqual(history["predictions"][0]["votes"], [0, 3, 2])

        charts = self.client.get("/api/prediction-stats").get_json()
        self.assertEqual(charts["total"], 2)
        self.assertEqual(sum(item["count"] for item in charts["categories"]), 2)
        self.assertEqual(sum(item["count"] for item in charts["by_department"]), 2)
        self.assertEqual(sum(item["count"] for item in charts["depth_distribution"]), 2)
        self.assertEqual(len(charts["magnitude_depth"]), 2)

    def test_predictions_can_be_deleted_individually_or_all_at_once(self):
        prediction_ids = []
        for department in ("Lima", "Cusco"):
            response = self.client.post("/api/predict", json={
                "department": department,
                "magnitude": 5.2,
                "depth": 35,
                "distance": 25,
                "k": 5,
            })
            prediction_ids.append(response.get_json()["record_id"])

        deleted_one = self.client.delete(f"/api/predictions/{prediction_ids[0]}")
        self.assertEqual(deleted_one.get_json()["deleted"], 1)
        self.assertEqual(self.client.get("/api/predictions").get_json()["total"], 1)
        self.assertEqual(self.client.delete("/api/predictions/999").status_code, 404)

        deleted_all = self.client.delete("/api/predictions")
        self.assertEqual(deleted_all.get_json()["deleted"], 1)
        self.assertEqual(self.client.get("/api/predictions").get_json()["total"], 0)
        self.assertEqual(self.client.get("/api/prediction-stats").get_json()["total"], 0)

        next_prediction = self.client.post("/api/predict", json={
            "department": "Lima",
            "magnitude": 5.2,
            "depth": 35,
            "distance": 25,
            "k": 5,
        })
        self.assertGreater(next_prediction.get_json()["record_id"], max(prediction_ids))

    def test_stats_charts_are_derived_from_event_data(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        stats = response.get_json()

        self.assertEqual(sum(item["count"] for item in stats["depth_distribution"]), len(MAP_EVENTS))
        self.assertEqual(
            stats["magnitude_by_department"],
            sorted(
                [
                    {"name": event["name"], "magnitude": event["magnitude"]}
                    for event in MAP_EVENTS
                ],
                key=lambda item: item["magnitude"],
                reverse=True,
            ),
        )
        self.assertEqual(len(stats["magnitude_depth"]), len(MAP_EVENTS))


if __name__ == "__main__":
    unittest.main()
