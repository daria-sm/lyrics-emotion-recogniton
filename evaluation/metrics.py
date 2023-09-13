import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def print_pandas_accuracy(df: pd.DataFrame):
    accuracy = (df['Predicted'] == df['Actual']).mean() * 100

    # Print the accuracy
    print(f"Accuracy: {accuracy:.2f}% for a dataset of {len(df)} ")


def print_pandas_precision(df: pd.DataFrame):
    # Extract predicted and actual labels from the DataFrame
    y_pred = df['Predicted']
    y_true = df['Actual']
    # Calculate precision using precision_score
    precision = precision_score(y_true, y_pred, average='binary', pos_label='Q1')

    # Print the precision
    print(f"Precision: {precision:.2f}")


def print_pandas_recall(df: pd.DataFrame):
    # Extract predicted and actual labels from the DataFrame
    y_pred = df['Predicted']
    y_true = df['Actual']

    # Calculate recall using recall_score
    recall = recall_score(y_true, y_pred, average='binary', pos_label='Q1')

    # Print the recall
    print(f"Recall: {recall:.2f}")


def print_pandas_f1_score(df: pd.DataFrame):
    # Extract predicted and actual labels from the DataFrame
    y_pred = df['Predicted']
    y_true = df['Actual']
    f1 = f1_score(y_true, y_pred, average='binary', pos_label='Q1')

    # Print the F1 score
    print(f"F1 score: {f1:.2f}")


def print_pandas_precision_non_binary(df: pd.DataFrame):
    # Extract predicted and actual labels from the DataFrame
    y_pred = df['Predicted']
    y_true = df['Actual']
    # Calculate precision using precision_score
    precision = precision_score(y_true, y_pred, average='macro')
    # Print the precision
    print(f"Precision: {precision:.2f}")


def print_pandas_recall_non_binary(df: pd.DataFrame):
    # Extract predicted and actual labels from the DataFrame
    y_pred = df['Predicted']
    y_true = df['Actual']

    # Calculate recall using recall_score
    recall = recall_score(y_true, y_pred, average='macro')

    # Print the recall
    print(f"Recall: {recall:.2f}")


def print_pandas_f1_score_non_binary(df: pd.DataFrame):
    # Extract predicted and actual labels from the DataFrame
    y_pred = df['Predicted']
    y_true = df['Actual']
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # Print the F1 score
    print(f"F1 score: {f1:.2f}")
