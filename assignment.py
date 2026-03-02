"""
==============================================================================
Sentiment Classification on PHM Tweet Dataset
Models:
  1. Stacked LSTM   (2 LSTM layers)
  2. Bidirectional LSTM (2 Bi-LSTM layers)

Run this file first. It trains both models and saves:
  - results/stacked_history.png
  - results/bi_history.png
  - results/stacked_cm.png
  - results/bi_cm.png
  - results/metrics.json

Then run generate_report.py to compile the PDF report.
==============================================================================
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os, json
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for saving files
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Bidirectional,
                                     Dense, Dropout, SpatialDropout1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ─── Output directory ────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

# ─── 0. CONFIG ───────────────────────────────────────────────────────────────
USE_GLOVE  = False
GLOVE_PATH = "glove.twitter.27B.50d.txt"

# ─── 1. DATA PREPARATION ────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)

NEGATIONS = {"not", "no", "never", "nor", "neither", "without",
             "cannot", "cant", "wont", "dont", "doesnt", "didnt",
             "wasn", "weren", "isn", "aren", "hasn", "haven", "hadn"}
english_stops = set(stopwords.words('english')) - NEGATIONS

train_df = pd.read_csv('phm_train.csv')
test_df  = pd.read_csv('phm_test.csv')

print(f"Train samples : {len(train_df)}")
print(f"Test  samples : {len(test_df)}")
print(f"Label dist (train):\n{train_df['label'].value_counts()}\n")

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in english_stops and len(w) > 1]
    return " ".join(tokens)

train_df['clean'] = train_df['tweet'].apply(clean)
test_df['clean']  = test_df['tweet'].apply(clean)

x_train, x_val, y_train, y_val = train_test_split(
    train_df['clean'], train_df['label'],
    test_size=0.2, random_state=42
)

# ─── 2. TOKENIZATION & PADDING ──────────────────────────────────────────────
token = Tokenizer(lower=True)
token.fit_on_texts(x_train)

lengths = [len(t.split()) for t in x_train]
max_len = int(np.percentile(lengths, 90))
total_words = len(token.word_index) + 1

print(f"Max sequence length (90th pct) : {max_len}")
print(f"Vocabulary size                : {total_words}\n")

def seq_and_pad(data):
    return pad_sequences(
        token.texts_to_sequences(data),
        maxlen=max_len, padding='post', truncating='post'
    )

x_train_pad = seq_and_pad(x_train)
x_val_pad   = seq_and_pad(x_val)
x_test_pad  = seq_and_pad(test_df['clean'])

# ─── 3. GLOVE (optional) ────────────────────────────────────────────────────
EMBED_DIM = 50 if USE_GLOVE else 32

def load_glove(path, word_index, dim):
    print(f"Loading GloVe from {path} ...")
    embeddings = np.zeros((len(word_index) + 1, dim))
    found = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            if word in word_index:
                embeddings[word_index[word]] = np.array(vals[1:], dtype='float32')
                found += 1
    print(f"GloVe: {found}/{len(word_index)} vocab words matched.")
    return embeddings

if USE_GLOVE:
    glove_matrix = load_glove(GLOVE_PATH, token.word_index, EMBED_DIM)

def make_embedding_layer():
    if USE_GLOVE:
        return Embedding(total_words, EMBED_DIM,
                         weights=[glove_matrix],
                         input_length=max_len, trainable=False)
    return Embedding(total_words, EMBED_DIM, input_length=max_len)

# ─── 4. HYPERPARAMETERS ──────────────────────────────────────────────────────
LSTM_OUT   = 64
DROPOUT    = 0.5
REC_DROP   = 0.3
EPOCHS     = 20
BATCH_SIZE = 128
REG        = l2(1e-4)

# ─── 5. MODEL DEFINITIONS ────────────────────────────────────────────────────
def build_stacked_lstm():
    model = Sequential([
        make_embedding_layer(),
        SpatialDropout1D(0.4),
        LSTM(LSTM_OUT, return_sequences=True,
             dropout=DROPOUT, recurrent_dropout=REC_DROP,
             kernel_regularizer=REG, recurrent_regularizer=REG),
        LSTM(LSTM_OUT // 2,
             dropout=DROPOUT, recurrent_dropout=REC_DROP,
             kernel_regularizer=REG, recurrent_regularizer=REG),
        Dense(32, activation='relu', kernel_regularizer=REG)/kjhjmlogdzzb ,
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bi_lstm():
    model = Sequential([
        make_embedding_layer(),
        SpatialDropout1D(0.4),
        Bidirectional(LSTM(LSTM_OUT, return_sequences=True,
                           dropout=DROPOUT, recurrent_dropout=REC_DROP,
                           kernel_regularizer=REG, recurrent_regularizer=REG)),
        Bidirectional(LSTM(LSTM_OUT // 2,
                           dropout=DROPOUT, recurrent_dropout=REC_DROP,
                           kernel_regularizer=REG, recurrent_regularizer=REG)),
        Dense(32, activation='relu', kernel_regularizer=REG),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ─── 6. CALLBACKS ────────────────────────────────────────────────────────────
def get_callbacks():
    es = EarlyStopping(monitor='val_loss', patience=5,
                       restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=2, min_lr=1e-6, verbose=1)
    return [es, rlr]

# ─── 7. TRAINING ─────────────────────────────────────────────────────────────
print("="*60)
print("Training Model 1: Stacked LSTM")
print("="*60)
stacked_model = build_stacked_lstm()
stacked_model.summary()
h_stacked = stacked_model.fit(
    x_train_pad, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val_pad, y_val),
    callbacks=get_callbacks(), verbose=1
)

print("\n" + "="*60)
print("Training Model 2: Bidirectional LSTM")
print("="*60)
bi_model = build_bi_lstm()
bi_model.summary()
h_bi = bi_model.fit(
    x_train_pad, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val_pad, y_val),
    callbacks=get_callbacks(), verbose=1
)

# ─── 8. SAVE TRAINING HISTORY PLOTS ─────────────────────────────────────────
def save_history_plot(history, title, filepath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1.plot(history.history['accuracy'],     label='Train Acc',  linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Acc',    linewidth=2, linestyle='--')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss',   linewidth=2, linestyle='--')
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

save_history_plot(h_stacked, 'Stacked LSTM (2 Layers) — Training History',
                  'results/stacked_history.png')
save_history_plot(h_bi,      'Bidirectional LSTM (2 Layers) — Training History',
                  'results/bi_history.png')

# ─── 9. EVALUATION & SAVE CONFUSION MATRICES ─────────────────────────────────
metrics_store = {}

def evaluate_and_save(model, name, key):
    y_pred_prob = model.predict(x_test_pad, verbose=0)
    y_pred      = (y_pred_prob >= 0.5).astype(int).flatten()
    y_true      = test_df['label'].values

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    correct = np.sum(y_pred == y_true)
    total   = len(y_pred)

    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print('='*60)
    print(f"Correct   : {correct} / {total}")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred))

    # Confusion matrix image
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                annot_kws={"size": 14})
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path = f'results/{key}_cm.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    # Store for JSON
    metrics_store[name] = {
        'correct': int(correct),
        'total': int(total),
        'accuracy': round(acc * 100, 2),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'epochs_trained': len(h_stacked.history['loss']) if key == 'stacked'
                          else len(h_bi.history['loss']),
        'classification_report': classification_report(y_true, y_pred)
    }

evaluate_and_save(stacked_model, 'Stacked LSTM (2 Layers)',      'stacked')
evaluate_and_save(bi_model,      'Bidirectional LSTM (2 Layers)', 'bi')

# ─── 10. SAVE METRICS JSON ───────────────────────────────────────────────────
metrics_store['dataset'] = {
    'train_samples': len(train_df),
    'test_samples' : len(test_df),
    'vocab_size'   : total_words,
    'max_seq_len'  : max_len,
    'embed_dim'    : EMBED_DIM,
    'glove_used'   : USE_GLOVE
}
metrics_store['hyperparameters'] = {
    'lstm_units'       : LSTM_OUT,
    'dropout'          : DROPOUT,
    'recurrent_dropout': REC_DROP,
    'batch_size'       : BATCH_SIZE,
    'max_epochs'       : EPOCHS,
    'l2_reg'           : 1e-4
}

with open('results/metrics.json', 'w') as f:
    json.dump(metrics_store, f, indent=2)

print("\nAll results saved to results/")
print("Now run:  python generate_report.py")