# Figures

PNG copies of every figure, so they can be viewed without opening the PDF or
running the notebook. The originals are vector PDFs in `report/figures/` (used
by the LaTeX source) and `results/figures/` (written by the notebook).

## Used in the report, in order of appearance

| File | Report | Shows |
|---|---|---|
| `fig01_network_pipeline.png` | Fig. 1 | Layer-by-layer tensor shapes through the CNN |
| `fig02_convolution.png` | Fig. 2 | A convolution step, and the effect of padding and stride |
| `fig03_pooling_and_head.png` | Fig. 3 | Max pooling, and the dense classification head |
| `fig04_why_cnns_win.png` | Fig. 4 | Parameter sharing, receptive fields, translation equivariance |
| `fig05_loss_curve_regimes.png` | Fig. 5 | Underfitting, good fit and overfitting as loss curves |
| `fig06_leakage_pairs.png` | Fig. 6 | Scans found in both `Training/` and `Testing/` |
| `fig07_learning_curves.png` | Fig. 7 | Training vs validation loss for all three runs |
| `fig08_confusion_matrix.png` | Fig. 8 | Confusion matrix for the augmented CNN |
| `fig09_misclassified.png` | Fig. 9 | The eight most confident mistakes |

Figures 1–5 are illustrations drawn for the theory sections. Figures 6–9 are
produced from the data and the trained models.

## Other notebook output

| File | Shows |
|---|---|
| `out_sample_images.png` | Four examples per class after preprocessing |
| `out_predictions.png` | Twelve test scans with predicted class and confidence |
| `out_predictions_wrong.png` | The same, restricted to misclassifications |

## The two worth looking at first

`fig06_leakage_pairs.png` — the top row is from `Testing/`, the bottom row from
`Training/`. They are the same scans. This is what a byte-level duplicate check
missed and what takes the honest test accuracy from 0.912 down to 0.902.

`fig09_misclassified.png` — six of the eight most confident errors are gliomas
called *notumor* at 100% confidence, several with a clearly visible lesion. High
confidence on an obvious lesion suggests mislabelled source data rather than a
weak model.
