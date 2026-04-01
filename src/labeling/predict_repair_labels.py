import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# =========================
# PATHS
# =========================
TRAIN_PATHS = [
    "data/photochat/train_json/train_00.json",
    "data/photochat/train_json/train_01.json",
]

PRED_PATH = "data/photochat/train_json/train_03.json"
OUTPUT_JSON = "data/photochat/predicted_train03.json"


# =========================
# DATA EXTRACTION
# =========================
def collect_labeled_examples(path):
    """
    Collect labeled P2 turns for training.

    Rules applied:
    - only user_id == 1
    - only before first share_photo == True
    - only turns that already have a label
    - only keep rows where previous P1 message exists
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for conv_idx, conv in enumerate(data):
        photo_id = conv.get("photo_id", None)
        dialogue = conv.get("dialogue", [])

        stopped = False

        for turn_idx, turn in enumerate(dialogue):
            if stopped:
                break

            if turn.get("share_photo") is True:
                stopped = True
                break

            if turn.get("user_id") != 1:
                continue

            if "label" not in turn:
                continue

            prev_p1_msg = None
            for j in range(turn_idx - 1, -1, -1):
                prev_turn = dialogue[j]
                if prev_turn.get("user_id") == 0:
                    prev_p1_msg = prev_turn.get("message", "").strip()
                    break

            if not prev_p1_msg:
                continue

            current_p2_msg = turn.get("message", "").strip()
            label = int(turn.get("label"))

            model_input = f"Prev_P1: {prev_p1_msg} [SEP] Current_P2: {current_p2_msg}"

            rows.append(
                {
                    "photo_id": photo_id,
                    "conv_idx": conv_idx,
                    "turn_idx": turn_idx,
                    "prev_p1_message": prev_p1_msg,
                    "current_p2_message": current_p2_msg,
                    "model_input": model_input,
                    "label": label,
                }
            )

    return rows


def collect_unlabeled_examples(path):
    """
    Collect unlabeled P2 turns for prediction.

    Rules applied:
    - only user_id == 1
    - only before first share_photo == True
    - only turns that do NOT already have a label
    - only keep rows where previous P1 message exists
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for conv_idx, conv in enumerate(data):
        photo_id = conv.get("photo_id", None)
        dialogue = conv.get("dialogue", [])

        stopped = False

        for turn_idx, turn in enumerate(dialogue):
            if stopped:
                break

            if turn.get("share_photo") is True:
                stopped = True
                break

            if turn.get("user_id") != 1:
                continue

            if "label" in turn:
                continue

            prev_p1_msg = None
            for j in range(turn_idx - 1, -1, -1):
                prev_turn = dialogue[j]
                if prev_turn.get("user_id") == 0:
                    prev_p1_msg = prev_turn.get("message", "").strip()
                    break

            if not prev_p1_msg:
                continue

            current_p2_msg = turn.get("message", "").strip()
            model_input = f"Prev_P1: {prev_p1_msg} [SEP] Current_P2: {current_p2_msg}"

            rows.append(
                {
                    "photo_id": photo_id,
                    "conv_idx": conv_idx,
                    "turn_idx": turn_idx,
                    "prev_p1_message": prev_p1_msg,
                    "current_p2_message": current_p2_msg,
                    "model_input": model_input,
                }
            )

    return rows


# =========================
# MODEL TRAINING
# =========================
def train_model(train_df):
    X = train_df["model_input"].tolist()
    y = train_df["label"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)

    print("\nValidation Report:")
    print(classification_report(y_val, val_pred, digits=4))

    return model, vectorizer


# =========================
# INSERT PREDICTIONS INTO JSON
# =========================
def insert_predictions_into_json(pred_df, input_path, output_path):
    """
    Create a new JSON file in the same format as the input file,
    but add predicted labels to the selected P2 turns.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred_lookup = {
        (int(row["conv_idx"]), int(row["turn_idx"])): int(row["predicted_label"])
        for _, row in pred_df.iterrows()
    }

    inserted_count = 0

    for conv_idx, conv in enumerate(data):
        dialogue = conv.get("dialogue", [])
        stopped = False

        for turn_idx, turn in enumerate(dialogue):
            if stopped:
                break

            if turn.get("share_photo") is True:
                stopped = True
                break

            if turn.get("user_id") != 1:
                continue

            if "label" in turn:
                continue

            key = (conv_idx, turn_idx)
            if key in pred_lookup:
                turn["label"] = pred_lookup[key]
                inserted_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved JSON with predictions to: {output_path}")
    print(f"Inserted predicted labels: {inserted_count}")


# =========================
# MAIN
# =========================
def main():
    print("Loading training data...")

    train_rows = []
    for path in TRAIN_PATHS:
        rows = collect_labeled_examples(path)
        print(f"{path}: usable labeled rows = {len(rows)}")
        train_rows.extend(rows)

    train_df = pd.DataFrame(train_rows)

    if train_df.empty:
        raise ValueError("No labeled training rows found.")

    print("\nTotal usable training rows:", len(train_df))
    print("\nTraining label distribution:")
    print(train_df["label"].value_counts().sort_index())

    model, vectorizer = train_model(train_df)

    print("\nLoading prediction data...")
    pred_rows = collect_unlabeled_examples(PRED_PATH)
    pred_df = pd.DataFrame(pred_rows)

    if pred_df.empty:
        raise ValueError("No unlabeled prediction rows found.")

    print("Usable prediction rows:", len(pred_df))

    X_pred = vectorizer.transform(pred_df["model_input"].tolist())
    pred_labels = model.predict(X_pred)

    pred_df["predicted_label"] = pred_labels

    print("\nFirst 10 predictions:")
    print(
        pred_df[
            [
                "photo_id",
                "conv_idx",
                "turn_idx",
                "prev_p1_message",
                "current_p2_message",
                "predicted_label",
            ]
        ].head(10).to_string(index=False)
    )

    insert_predictions_into_json(
        pred_df=pred_df,
        input_path=PRED_PATH,
        output_path=OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()