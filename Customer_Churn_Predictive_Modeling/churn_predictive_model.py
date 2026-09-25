# ============================================================
# CUSTOMER CHURN PREDICTIVE MODELLING USING ANN
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 2. DEFINE FILE PATHS
# ============================================================

DATA_PATH = r"Data\customer_churn_preprocessed_Data.csv"

RESULTS_DIR = "Results"
MODELS_DIR = "Models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. CHECK DATASET INFORMATION
# ============================================================

print("\nDataset information:")
print(df.info())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 5. CHECK TARGET VARIABLE
# ============================================================

print("\nUnique values in each column:")

for column in df.columns:
    print(column, ":", df[column].unique()[:10])


# ============================================================
# 6. DEFINE TARGET VARIABLE
# ============================================================

TARGET_COLUMN = "Churn"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found. "
        f"Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# 7. ENCODE TARGET VARIABLE
# ============================================================

print("\nTarget variable before encoding:")
print(df[TARGET_COLUMN].value_counts())

if df[TARGET_COLUMN].dtype == "object":

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0
        })
    )

if df[TARGET_COLUMN].isnull().any():
    raise ValueError(
        "The Churn column contains values that could not be encoded."
    )

print("\nTarget variable after encoding:")
print(df[TARGET_COLUMN].value_counts())


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# ============================================================
# 9. HANDLE NON-NUMERIC FEATURES
# ============================================================

print("\nChecking feature data types:")

print(X.dtypes)

non_numeric_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNon-numeric columns:")

if len(non_numeric_columns) == 0:
    print("None")
else:
    print(non_numeric_columns)


# ============================================================
# 10. ONE-HOT ENCODE CATEGORICAL VARIABLES
# ============================================================

if len(non_numeric_columns) > 0:

    X = pd.get_dummies(
        X,
        columns=non_numeric_columns,
        drop_first=True
    )

print("\nFeatures after encoding:")
print("Number of features:", X.shape[1])


# ============================================================
# 11. CONVERT FEATURES TO NUMERIC
# ============================================================

X = X.astype(float)

print("\nFinal feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 12. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 13. FEATURE SCALING
# ============================================================

# The dataset supplied by the Data Engineer already contains
# preprocessed/scaled numerical variables.
#
# Therefore, we do not apply another StandardScaler here.
#
# This avoids scaling already-scaled data unnecessarily.


# ============================================================
# 14. CONVERT DATA TO NUMPY ARRAYS
# ============================================================

X_train = np.asarray(X_train).astype("float32")
X_test = np.asarray(X_test).astype("float32")

y_train = np.asarray(y_train).astype("float32")
y_test = np.asarray(y_test).astype("float32")


# ============================================================
# 15. DEFINE ANN ARCHITECTURE
# ============================================================

# ============================================================
# 15. DEFINE ANN ARCHITECTURE
# ============================================================

input_features = X_train.shape[1]

print("\nNumber of input features:", input_features)

model = Sequential([
    
    Input(shape=(input_features,)),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

# ============================================================
# 16. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\nANN Model Architecture:")

model.summary()


# ============================================================
# 17. COMPILE ANN
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 18. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ============================================================
# 19. TRAIN ANN MODEL
# ============================================================



# Calculate class weights to handle class imbalance
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:")
print(class_weights)


print("\nTraining ANN model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# 20. MODEL EVALUATION ON TEST DATA
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 21. MAKE PREDICTIONS
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_pred = (y_probability >= 0.50).astype(int)


# ============================================================
# 22. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 23. DISPLAY PERFORMANCE METRICS
# ============================================================

print("\n==============================================")
print("ANN PERFORMANCE RESULTS")
print("==============================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 24. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
)

print(report)


# ============================================================
# 25. SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write("ANN CUSTOMER CHURN PREDICTIVE MODEL\n")
    file.write("====================================\n\n")

    file.write(f"Accuracy  : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"F1 Score  : {f1:.4f}\n")
    file.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")

    file.write("Classification Report\n")
    file.write("---------------------\n")
    file.write(report)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 27. TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("ANN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "training_validation_accuracy.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 28. TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "training_validation_loss.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 29. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ANN (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - ANN Customer Churn Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 30. SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "customer_churn_ann_model.keras"
)

model.save(MODEL_PATH)

print("\nANN model saved to:")
print(MODEL_PATH)


# ============================================================
# 31. SAVE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({
    "Actual_Churn": y_test.astype(int),
    "Predicted_Churn": y_pred,
    "Churn_Probability": y_probability
})

prediction_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "ann_predictions.csv"
    ),
    index=False
)

print("\nPredictions saved to:")
print(
    os.path.join(
        RESULTS_DIR,
        "ann_predictions.csv"
    )
)


# ============================================================
# 32. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics.to_csv(
    os.path.join(
        RESULTS_DIR,
        "ann_model_metrics.csv"
    ),
    index=False
)


# ============================================================
# 33. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("PREDICTIVE MODELLING COMPLETED")
print("==============================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nResults saved in:")
print(RESULTS_DIR)

print("\nModel saved in:")
print(MODELS_DIR)

print("\nDone.")




