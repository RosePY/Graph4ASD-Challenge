import os
from datetime import datetime
from pathlib import Path
from .hidden_labels_reader import read_hidden_labels
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"

def read_submission_files():
    files = os.listdir(SUBMISSIONS_DIR)
    return [f"{SUBMISSIONS_DIR}/{f}" for f in files if f.endswith('.csv')]


def _team_name_from_path(path: Path) -> str:
    parts = path.parts
    if "inbox" in parts:
        try:
            inbox_idx = parts.index("inbox")
            team = parts[inbox_idx + 1]
            run_id = parts[inbox_idx + 2]
            return f"{team}/{run_id}"
        except (IndexError, ValueError):
            return path.stem
    return path.stem


def calculate_scores(submission_path: Path):
    submission_path = Path(submission_path).resolve()

    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    labels_df = read_hidden_labels()
    if labels_df is None:
        raise FileNotFoundError("Labels file not found. Have you added TEST_LABELS_CSV to your .env file or secrets?")
    
    submission_df = pd.read_csv(submission_path)

    if "id" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("Labels file must contain 'id' and 'label' columns.")

    prediction_col = "label" if "label" in submission_df.columns else "label"
    if "id" not in submission_df.columns or prediction_col not in submission_df.columns:
        raise ValueError("Submission file must contain 'filename' and 'prediction' columns.")

    merged = labels_df.merge(
        submission_df[["id", prediction_col]],
        on="id",
        how="outer",
        indicator=True,
    )
    missing_in_submission = merged[merged["_merge"] == "left_only"]["id"].tolist()
    missing_in_labels = merged[merged["_merge"] == "right_only"]["id"].tolist()
    if missing_in_submission or missing_in_labels:
        raise ValueError(
            "Filename mismatch between labels and submission. "
            f"Missing in submission: {missing_in_submission[:5]}. "
            f"Missing in labels: {missing_in_labels[:5]}."
        )

    y_true = pd.to_numeric(merged["label"], errors="coerce")
    y_pred = pd.to_numeric(merged[prediction_col], errors="coerce")
    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("Non-numeric targets or predictions detected.")

    validation_accuracy = accuracy_score(y_true, y_pred)
    validation_f1_score = f1_score(y_true, y_pred, average="macro")
    return {
        "validation_accuracy": float(validation_accuracy),
        "validation_f1_score": float(validation_f1_score),
    }


def get_leaderboard_data():
    files = read_submission_files()
    scores = []

    for submission_path in files:
        submission_path = Path(submission_path)
        team_name = _team_name_from_path(submission_path)
        timestamp = datetime.fromtimestamp(submission_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        team_scores = calculate_scores(submission_path)
        scores.append(
            {
                "team_name": team_name,
                **team_scores,
                "timestamp": timestamp,
            }
        )

    scores.sort(key=lambda x: x["validation_f1_score"], reverse=True)
    return scores

if __name__ == "__main__":
    leaderboard_data = get_leaderboard_data()

    for team_submission in leaderboard_data:
        print(f"Team: {team_submission['team_name']}")
        print(f"Validation F1 Score: {team_submission['validation_f1_score'] * 100:.2f}%")
        print(f"Validation Accuracy: {team_submission['validation_accuracy'] * 100:.2f}%")
        print(f"Timestamp: {team_submission['timestamp']}")
        print("-" * 50)
