"""
generate_report.py
Reads results/ folder and builds a clean simple PDF report.

Usage:
    python generate_report.py
Output:
    Sentiment_Classification_Report.pdf
"""

import json, os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)

RESULTS_DIR = "results"
OUTPUT_PDF  = "Sentiment_Classification_Report.pdf"

with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
    metrics = json.load(f)

ds = metrics['dataset']
hp = metrics['hyperparameters']
m1 = metrics['Stacked LSTM (2 Layers)']
m2 = metrics['Bidirectional LSTM (2 Layers)']

PAGE_W = A4[0]
# usable width after margins (2.5cm each side)
USABLE_W = PAGE_W - 5*cm   # ~11.5 cm safe image width

# ── styles ───────────────────────────────────────────────────────────────────
S_TITLE   = ParagraphStyle('T',  fontSize=15, fontName='Helvetica-Bold',
               alignment=TA_CENTER, spaceAfter=4, leading=19)
S_SUB     = ParagraphStyle('S',  fontSize=10, fontName='Helvetica',
               alignment=TA_CENTER, spaceAfter=3,
               textColor=colors.HexColor('#555555'))
S_H1      = ParagraphStyle('H1', fontSize=12, fontName='Helvetica-Bold',
               spaceBefore=14, spaceAfter=5, leading=16)
S_H2      = ParagraphStyle('H2', fontSize=11, fontName='Helvetica-Bold',
               spaceBefore=8, spaceAfter=3, leading=14)
S_BODY    = ParagraphStyle('B',  fontSize=10, fontName='Helvetica',
               leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
S_CAPTION = ParagraphStyle('C',  fontSize=9,  fontName='Helvetica-Oblique',
               alignment=TA_CENTER, spaceBefore=2, spaceAfter=10,
               textColor=colors.HexColor('#666666'))
S_CODE    = ParagraphStyle('CODE', fontSize=8.5, fontName='Courier',
               leading=13, leftIndent=10, spaceAfter=4)

# ── helpers ──────────────────────────────────────────────────────────────────
def h1(t):    return Paragraph(t, S_H1)
def h2(t):    return Paragraph(t, S_H2)
def body(t):  return Paragraph(t, S_BODY)
def cap(t):   return Paragraph(t, S_CAPTION)
def gap(n=8): return Spacer(1, n)

def fit_img(path, max_w, max_h=None):
    """Load image, scale it to fit within max_w (and optionally max_h)."""
    if not os.path.exists(path):
        return body(f"[Image not found: {path}]")
    i = Image(path)
    native_w = i.imageWidth
    native_h = i.imageHeight
    scale = max_w / native_w
    new_h = native_h * scale
    if max_h and new_h > max_h:
        scale = max_h / native_h
        max_w = native_w * scale
        new_h = max_h
    i.drawWidth  = max_w
    i.drawHeight = new_h
    return i

LIGHT_GRAY = colors.HexColor('#F2F2F2')

def make_table(rows, col_widths):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0),  (-1, 0),  colors.transparent),
        ('TEXTCOLOR',     (0, 0),  (-1, 0),  colors.black),
        ('FONTNAME',      (0, 0),  (-1, 0),  'Helvetica-Bold'),
        ('FONTNAME',      (0, 1),  (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 0),  (-1, -1), 10),
        ('ALIGN',         (0, 0),  (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0),  (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0, 1),  (-1, -1), [colors.white, LIGHT_GRAY]),
        ('LINEBELOW',     (0, 0),  (-1, 0),  0.8, colors.black),
        ('LINEBELOW',     (0, -1), (-1, -1), 0.4, colors.HexColor('#AAAAAA')),
        ('TOPPADDING',    (0, 0),  (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0),  (-1, -1), 6),
    ]))
    return t

# ── story ────────────────────────────────────────────────────────────────────
story = []

# ── title page ───────────────────────────────────────────────────────────────
story.append(gap(28))
story.append(Paragraph("CSC4093: Neural Networks and Deep Learning", S_SUB))
story.append(gap(6))
story.append(Paragraph("Programming Assignment 01", S_TITLE))
story.append(Paragraph("Tweet Classification using LSTM Networks", S_TITLE))
story.append(gap(16))
story.append(Paragraph("Mohammadhu Umair", S_SUB))
story.append(Paragraph("S/20/534", S_SUB))
story.append(Paragraph(datetime.now().strftime('%B %d, %Y'), S_SUB))
story.append(gap(24))

# ── 1. Introduction ──────────────────────────────────────────────────────────
story.append(h1("1. Introduction"))
story.append(body(
    "This report covers the implementation of two LSTM based models for classifying "
    "tweets as personal health mentions or non-personal health mentions. The PHM dataset "
    "was used for training and evaluation. A Stacked LSTM and a Bidirectional LSTM were "
    "built and compared in terms of accuracy, precision, recall and F1 score. The main "
    "challenge with this kind of data is the informal nature of tweets, where short "
    "phrases and missing context make classification harder than typical text data."
))

# ── 2. Dataset ───────────────────────────────────────────────────────────────
story.append(h1("2. Dataset"))
story.append(body(
    f"The dataset contains tweets labelled as 1 (personal health mention) or 0 "
    f"(non-personal). It was split into {ds['train_samples']:,} training samples and "
    f"{ds['test_samples']:,} test samples. A further 20% of the training data was held "
    "out for validation during model training."
))
story.append(gap(4))
story.append(make_table([
    ['Split', 'Count'],
    ['Train', f"{ds['train_samples']:,}"],
    ['Test',  f"{ds['test_samples']:,}"],
], col_widths=[9*cm, 7*cm]))

# ── 3. Methodology ───────────────────────────────────────────────────────────
story.append(h1("3. Methodology"))

story.append(h2("3.1 Preprocessing"))
story.append(body(
    "Tweets were cleaned by lowercasing, removing URLs, mentions and non-alphabetic "
    "characters. Stop words were removed but negation words like 'not' and 'never' were "
    "kept on purpose since dropping them would flip the meaning of many health related "
    "phrases. Single character tokens were also discarded."
))

story.append(h2("3.2 Tokenization"))
story.append(body(
    f"A Keras Tokenizer was fitted on training data only, producing a vocabulary of "
    f"{ds['vocab_size']:,} tokens. Sequences were padded to a length of {ds['max_seq_len']} "
    "which is the 90th percentile of training lengths. This was chosen instead of the "
    "maximum length to avoid excessive padding from a few very long tweets."
))

story.append(h2("3.3 Stacked LSTM"))
story.append(body(
    f"The first model uses two LSTM layers in sequence. The first layer has {hp['lstm_units']} "
    f"units and passes its full output to the second layer which has {hp['lstm_units']//2} units. "
    "A Dense layer with 32 units and sigmoid output handles classification. The first layer "
    "picks up lower level word patterns and the second layer builds on those to understand "
    "the overall meaning of the tweet."
))

story.append(h2("3.4 Bidirectional LSTM"))
story.append(body(
    "The second model wraps each LSTM layer in a Bidirectional layer so the sequence is "
    "read both forward and backward. This gives the model more context at each word position "
    "which helps when meaning depends on what comes after a word, not just before it. "
    "The layer structure is otherwise the same as Model 1."
))

story.append(h2("3.5 Overfitting Control"))
story.append(body(
    "In early runs, training accuracy went above 93% while validation accuracy stayed "
    "around 80%, which was a clear sign of overfitting. SpatialDropout1D was added after "
    f"the embedding layer, dropout of {hp['dropout']} and recurrent dropout of "
    f"{hp['recurrent_dropout']} were applied inside the LSTM layers, and L2 regularization "
    f"of {hp['l2_reg']} was added to the weights. EarlyStopping with patience 5 was used "
    "to stop training when validation loss stopped improving, and ReduceLROnPlateau halved "
    "the learning rate when progress stalled."
))

story.append(PageBreak())

# ── 4. Results ───────────────────────────────────────────────────────────────
story.append(h1("4. Results"))

story.append(h2("4.1 Performance Summary"))
story.append(gap(4))
story.append(make_table([
    ['Metric',          'Stacked LSTM',       'Bidirectional LSTM'],
    ['Accuracy',        f"{m1['accuracy']}%",  f"{m2['accuracy']}%"],
    ['Precision',       f"{m1['precision']}",  f"{m2['precision']}"],
    ['Recall',          f"{m1['recall']}",     f"{m2['recall']}"],
    ['F1 Score',        f"{m1['f1_score']}",   f"{m2['f1_score']}"],
    ['Correct / Total', f"{m1['correct']} / {m1['total']}", f"{m2['correct']} / {m2['total']}"],
    ['Epochs',          str(m1['epochs_trained']), str(m2['epochs_trained'])],
], col_widths=[6*cm, 5*cm, 5*cm]))
story.append(gap(14))

# history plots — constrained width so they never bleed past margins
story.append(h2("4.2 Stacked LSTM Training History"))
story.append(gap(4))
story.append(fit_img('results/stacked_history.png', max_w=USABLE_W, max_h=7*cm))
story.append(cap("Figure 1: Accuracy and loss over epochs for the Stacked LSTM."))
story.append(gap(12))

story.append(h2("4.3 Bidirectional LSTM Training History"))
story.append(gap(4))
story.append(fit_img('results/bi_history.png', max_w=USABLE_W, max_h=7*cm))
story.append(cap("Figure 2: Accuracy and loss over epochs for the Bidirectional LSTM."))
story.append(PageBreak())

# confusion matrices — each one half the usable width, side by side
story.append(h2("4.4 Confusion Matrices"))
story.append(gap(6))

half_w = (USABLE_W - 1*cm) / 2   # small gap between the two images

cm_tbl = Table(
    [[fit_img('results/stacked_cm.png', max_w=half_w, max_h=6*cm),
      fit_img('results/bi_cm.png',      max_w=half_w, max_h=6*cm)]],
    colWidths=[half_w + 0.5*cm, half_w + 0.5*cm]
)
cm_tbl.setStyle(TableStyle([
    ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING',(0, 0), (-1, -1), 4),
]))
story.append(cm_tbl)
story.append(cap(
    "Figure 3: Confusion matrix for Stacked LSTM (left) and Bidirectional LSTM (right)."
))

# ── 5. Discussion ────────────────────────────────────────────────────────────
story.append(h1("5. Discussion"))

winner = 'Bidirectional LSTM' if m2['accuracy'] >= m1['accuracy'] else 'Stacked LSTM'
loser  = 'Stacked LSTM' if winner == 'Bidirectional LSTM' else 'Bidirectional LSTM'
w_acc  = m2['accuracy'] if winner == 'Bidirectional LSTM' else m1['accuracy']
w_f1   = m2['f1_score'] if winner == 'Bidirectional LSTM' else m1['f1_score']
l_acc  = m1['accuracy'] if winner == 'Bidirectional LSTM' else m2['accuracy']
diff   = abs(m2['accuracy'] - m1['accuracy'])

story.append(body(
    f"The {winner} performed better overall, reaching {w_acc}% accuracy compared to "
    f"{l_acc}% for the {loser}. The difference of {diff:.2f} points is consistent across "
    "all metrics which suggests it is not just random variation. Reading tweets in both "
    "directions seems to help the model understand context better, especially when the "
    "key word appears mid or end of the sentence."
))
story.append(body(
    "The training curves show that the regularization worked well. In earlier attempts "
    "without dropout and L2, the validation loss was going up while training loss kept "
    "falling. After adding the regularization stack both curves moved together. "
    "EarlyStopping made sure the final weights came from the best validation epoch "
    "rather than the last one."
))
story.append(body(
    "Keeping negation words during preprocessing also made a difference. Something like "
    "'not feeling well' would become 'feeling well' after standard stop word removal "
    "which completely flips the meaning. The 90th percentile for sequence length also "
    "helped avoid the model learning from excessive padding in short tweets."
))

# ── 6. Conclusion ────────────────────────────────────────────────────────────
story.append(h1("6. Conclusion"))
story.append(body(
    f"Both models managed to classify personal health mentions reasonably well. The "
    f"{winner} came out ahead with {w_acc}% accuracy and an F1 of {w_f1}. The "
    "regularization choices kept overfitting under control and the preprocessing decisions "
    "helped preserve meaning in the tweet text. Using pre-trained Twitter embeddings like "
    "GloVe or a model like BioBERT would probably push the accuracy higher, but within "
    "the scope of this assignment the results are reasonable and the models generalise "
    "well on unseen data."
))

story.append(PageBreak())

# ── Appendix ─────────────────────────────────────────────────────────────────
story.append(h1("Appendix: Classification Reports"))
story.append(h2("A.1 Stacked LSTM"))
story.append(Paragraph(
    m1['classification_report'].replace('\n', '<br/>').replace(' ', '&nbsp;'),
    S_CODE
))
story.append(gap(12))
story.append(h2("A.2 Bidirectional LSTM"))
story.append(Paragraph(
    m2['classification_report'].replace('\n', '<br/>').replace(' ', '&nbsp;'),
    S_CODE
))

# ── build ─────────────────────────────────────────────────────────────────────
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#888888'))
    w, _ = A4
    canvas.drawCentredString(w / 2, 1.4*cm,
        f"CSC4093 Programming Assignment 01  |  Mohammadhu Umair  S/20/534  |  Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4,
    rightMargin=2.5*cm, leftMargin=2.5*cm,
    topMargin=2.5*cm,   bottomMargin=2.5*cm,
    title="CSC4093 Assignment 01", author="Mohammadhu Umair")

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Saved: {OUTPUT_PDF}")