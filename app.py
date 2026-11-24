from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import sqlite3

from flask import Flask, jsonify, render_template, request
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config["DATABASE"] = str(Path(app.instance_path) / "predictions.sqlite3")

TRAINING_DATA = np.array([
    [3.8, 70, 180], [4.2, 50, 140], [4.5, 90, 110],
    [5.0, 40, 70], [5.4, 55, 45], [5.8, 30, 60],
    [6.4, 20, 35], [7.0, 25, 20], [7.8, 15, 25],
], dtype=float)
TRAINING_LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
CATEGORIES = ["Leve", "Moderada", "Fuerte"]

# Rangos unificados (donut, mapa y leyenda deben usar estos)
MAG_BINS = [
    {"label": "< 4.0", "min": 0.0, "max": 4.0, "color": "#6b9bd1"},
    {"label": "4.0–5.9", "min": 4.0, "max": 6.0, "color": "#e8b06b"},
    {"label": "≥ 6.0", "min": 6.0, "max": 99.0, "color": "#d9696f"},
]

MAP_EVENTS = [
    {"name": "Huanuco", "lat": -9.9306, "lng": -76.2422, "magnitude": 5.2, "depth": 35},
    {"name": "Lima", "lat": -12.0464, "lng": -77.0428, "magnitude": 6.1, "depth": 42},
    {"name": "Arequipa", "lat": -16.4090, "lng": -71.5375, "magnitude": 5.8, "depth": 68},
    {"name": "Ica", "lat": -14.0678, "lng": -75.7286, "magnitude": 7.1, "depth": 25},
    {"name": "Ancash", "lat": -9.5261, "lng": -77.5288, "magnitude": 6.4, "depth": 31},
    {"name": "Piura", "lat": -5.1945, "lng": -80.6328, "magnitude": 4.7, "depth": 55},
    {"name": "Cusco", "lat": -13.5319, "lng": -71.9675, "magnitude": 5.5, "depth": 48},
]


def build_pipeline(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


@lru_cache(maxsize=3)
def train_model(k):
    return build_pipeline(k).fit(TRAINING_DATA, TRAINING_LABELS)


@lru_cache(maxsize=3)
def model_performance(k):
    """Validación leave-one-out sobre los datos de entrenamiento."""
    preds = cross_val_predict(build_pipeline(k), TRAINING_DATA, TRAINING_LABELS, cv=LeaveOneOut())
    matrix = confusion_matrix(TRAINING_LABELS, preds, labels=[0, 1, 2])
    return {
        "labels": CATEGORIES,
        "matrix": matrix.tolist(),
        "total": int(matrix.sum()),
        "accuracy": round(accuracy_score(TRAINING_LABELS, preds) * 100, 1),
    }


def magnitude_distribution():
    total = len(MAP_EVENTS)
    result = []
    for b in MAG_BINS:
        count = sum(1 for e in MAP_EVENTS if b["min"] <= e["magnitude"] < b["max"])
        result.append({**b, "count": count, "pct": round(count / total * 100, 1)})
    return result


def depth_distribution():
    bins = [
        {"label": "0–70 km", "min": 0, "max": 70},
        {"label": "70–300 km", "min": 70, "max": 300},
        {"label": "300–700 km", "min": 300, "max": 701},
    ]
    return [
        {
            "label": item["label"],
            "count": sum(item["min"] <= event["depth"] < item["max"] for event in MAP_EVENTS),
        }
        for item in bins
    ]


def events_by_department():
    counts = {}
    for e in MAP_EVENTS:
        counts[e["name"]] = counts.get(e["name"], 0) + 1
    return sorted(
        ({"name": n, "count": c} for n, c in counts.items()),
        key=lambda d: d["count"],
        reverse=True,
    )


def magnitude_by_department():
    return sorted(
        ({"name": event["name"], "magnitude": event["magnitude"]} for event in MAP_EVENTS),
        key=lambda item: item["magnitude"],
        reverse=True,
    )


def build_metrics():
    return {
        "event_count": len(MAP_EVENTS),
        "max_magnitude": max(e["magnitude"] for e in MAP_EVENTS),
        "average_depth": round(sum(e["depth"] for e in MAP_EVENTS) / len(MAP_EVENTS)),
        "region_count": len({e["name"] for e in MAP_EVENTS}),
    }


@contextmanager
def prediction_database():
    database_path = Path(app.config["DATABASE"])
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                department TEXT NOT NULL,
                magnitude REAL NOT NULL,
                depth REAL NOT NULL,
                distance REAL NOT NULL,
                neighbors INTEGER NOT NULL,
                category TEXT NOT NULL,
                tied INTEGER NOT NULL,
                votes TEXT NOT NULL,
                percentages TEXT NOT NULL
            )
        """)
        connection.commit()
        yield connection
    finally:
        connection.close()


def save_prediction(department, magnitude, depth, distance, k, result):
    with prediction_database() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at, department, magnitude, depth, distance, neighbors,
                category, tied, votes, percentages
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                department,
                magnitude,
                depth,
                distance,
                k,
                result["category"],
                int(result["tied"]),
                json.dumps(result["votes"]),
                json.dumps(result["percentages"]),
            ),
        )
        connection.commit()
        return cursor.lastrowid


def recent_predictions(limit):
    with prediction_database() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, department, magnitude, depth, distance,
                   neighbors, category, tied, votes, percentages
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    return {
        "total": total,
        "predictions": [
            {
                **dict(row),
                "tied": bool(row["tied"]),
                "votes": json.loads(row["votes"]),
                "percentages": json.loads(row["percentages"]),
            }
            for row in rows
        ],
    }


def prediction_statistics():
    with prediction_database() as connection:
        total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        category_counts = {
            row["category"]: row["count"]
            for row in connection.execute(
                "SELECT category, COUNT(*) AS count FROM predictions GROUP BY category"
            ).fetchall()
        }
        by_department = connection.execute(
            """
            SELECT department AS name, COUNT(*) AS count
            FROM predictions
            GROUP BY department
            ORDER BY count DESC, department
            """
        ).fetchall()
        depth_counts = connection.execute(
            """
            SELECT
                SUM(CASE WHEN depth < 70 THEN 1 ELSE 0 END) AS shallow,
                SUM(CASE WHEN depth >= 70 AND depth < 300 THEN 1 ELSE 0 END) AS intermediate,
                SUM(CASE WHEN depth >= 300 THEN 1 ELSE 0 END) AS deep,
                MAX(depth) AS max_depth
            FROM predictions
            """
        ).fetchone()
        recent_rows = connection.execute(
            """
            SELECT id, department AS name, magnitude, depth, category
            FROM predictions
            ORDER BY id DESC
            LIMIT 50
            """
        ).fetchall()

    depth_axis_max = max(
        100,
        math.ceil((depth_counts["max_depth"] or 0) / 100) * 100,
    )
    categories = [
        {"label": category, "count": category_counts.get(category, 0), "color": color}
        for category, color in zip(CATEGORIES, ("#6b9bd1", "#e8b06b", "#d9696f"))
    ]
    return {
        "total": total,
        "categories": categories,
        "by_department": [dict(row) for row in by_department],
        "depth_distribution": [
            {"label": "0–70 km", "count": depth_counts["shallow"] or 0},
            {"label": "70–300 km", "count": depth_counts["intermediate"] or 0},
            {"label": "300–700 km", "count": depth_counts["deep"] or 0},
        ],
        "magnitude_depth": [dict(row) for row in reversed(recent_rows)],
        "depth_axis_max": depth_axis_max,
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        events=MAP_EVENTS,
        metrics=build_metrics(),
        distribution=magnitude_distribution(),
        by_department=events_by_department(),
        performance=model_performance(5),
        mag_bins=MAG_BINS,
    )


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        magnitude = float(data["magnitude"])
        depth = float(data["depth"])
        distance = float(data["distance"])
        k = int(data["k"])
        department = data.get("department", "Sin especificar")
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Parametros invalidos"}), 400

    if not isinstance(department, str) or not department.strip() or len(department) > 100:
        return jsonify({"error": "El departamento no es valido"}), 400
    department = department.strip()
    if not all(math.isfinite(v) for v in (magnitude, depth, distance)):
        return jsonify({"error": "Los parametros deben ser numeros finitos"}), 400
    if not (3 <= magnitude <= 9 and 0 <= depth <= 700 and 0 <= distance <= 1000):
        return jsonify({"error": "Los valores estan fuera de rango"}), 400
    if k not in (3, 5, 7):
        return jsonify({"error": "K debe ser 3, 5 o 7 para este conjunto de demostracion"}), 400

    model = train_model(k)
    point = np.array([[magnitude, depth, distance]])

    neighbor_ids = model.named_steps["knn"].kneighbors(
        model.named_steps["scaler"].transform(point),
        n_neighbors=k,
        return_distance=False,
    )[0].tolist()

    votes = [0, 0, 0]
    for nid in neighbor_ids:
        votes[int(TRAINING_LABELS[nid])] += 1

    highest = max(votes)
    winners = [i for i, v in enumerate(votes) if v == highest]
    # Desempate: ante empate se elige la clase más severa (más seguro en sismos)
    category = CATEGORIES[max(winners)]

    result = {
        "category": category,
        "tied": len(winners) > 1,
        "tied_categories": [CATEGORIES[i] for i in winners] if len(winners) > 1 else [],
        "percentages": [round(v / k * 100) for v in votes],
        "votes": votes,
        "neighbors": k,
    }
    try:
        result["record_id"] = save_prediction(
            department, magnitude, depth, distance, k, result
        )
    except sqlite3.Error:
        app.logger.exception("No se pudo guardar la estimacion sísmica")
        return jsonify({"error": "La estimacion se calculo, pero no se pudo guardar en el historial"}), 500
    return jsonify(result)


@app.get("/api/events")
def events():
    return jsonify(MAP_EVENTS)


@app.get("/api/predictions")
def predictions():
    limit = request.args.get("limit", 100, type=int)
    if limit is None or not 1 <= limit <= 500:
        return jsonify({"error": "El limite debe estar entre 1 y 500"}), 400
    return jsonify(recent_predictions(limit))


@app.get("/api/prediction-stats")
def prediction_stats():
    return jsonify(prediction_statistics())


@app.delete("/api/predictions")
def clear_predictions():
    with prediction_database() as connection:
        deleted = connection.execute("DELETE FROM predictions").rowcount
        connection.commit()
    return jsonify({"deleted": deleted})


@app.delete("/api/predictions/<int:prediction_id>")
def delete_prediction(prediction_id):
    with prediction_database() as connection:
        cursor = connection.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        connection.commit()
    if cursor.rowcount == 0:
        return jsonify({"error": "No se encontro esa estimacion"}), 404
    return jsonify({"deleted": cursor.rowcount, "id": prediction_id})


@app.get("/api/stats")
def stats():
    depth_axis_max = max(100, math.ceil(max(event["depth"] for event in MAP_EVENTS) / 100) * 100)
    return jsonify({
        "metrics": build_metrics(),
        "distribution": magnitude_distribution(),
        "by_department": events_by_department(),
        "depth_distribution": depth_distribution(),
        "magnitude_by_department": magnitude_by_department(),
        "magnitude_depth": [
            {"name": event["name"], "magnitude": event["magnitude"], "depth": event["depth"]}
            for event in MAP_EVENTS
        ],
        "depth_axis_max": depth_axis_max,
    })


@app.get("/api/performance")
def performance():
    k = request.args.get("k", 5, type=int)
    if k not in (3, 5, 7):
        return jsonify({"error": "K debe ser 3, 5 o 7"}), 400
    return jsonify(model_performance(k))


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)