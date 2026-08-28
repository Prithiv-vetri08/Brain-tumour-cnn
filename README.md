# Brain Tumour MRI Classification with CNNs

Exam 1 — Artificial Intelligence / Machine Learning (BST-AM)
Prithiv Ramvasan Vetri Selvan · 3121993
B.Eng. Mechatronic Systems Engineering, SRH Berlin University of Applied Sciences

Not a clinical tool. This is a coursework exercise in image classification.

---

## What this is

Four-class classification of brain MRI scans (glioma, meningioma, pituitary
tumour, no tumour). A fully connected baseline is compared against a CNN, and
the CNN is then compared against itself with and without data augmentation, so
each comparison changes one thing at a time.

## Results

| Model | Parameters | Test accuracy | Macro F1 |
|---|---|---|---|
| MLP baseline | 25,168,388 | 0.644 | 0.641 |
| Improved CNN | 1,290,404 | 0.862 | 0.860 |
| Improved CNN + augmentation | 1,290,404 | 0.912 | 0.910 |
| Improved CNN + augmentation, clean test subset | 1,290,404 | **0.902** | **0.866** |

The CNN gains 21.7 points over the MLP while using 19.5× fewer parameters.
Augmentation adds 5.0 points more.

## One thing to read before trusting the numbers

The test split shipped with the dataset is not clean. The same scans appear in
both `Training/` and `Testing/`, re-encoded, so a hash of the file bytes reports
no overlap while the decoded images match. Comparing pixels instead, 443 of the
1,600 test images (27.7%) have a near-identical counterpart in training — 333 of
them in the *notumor* class alone.

On the 1,157 genuinely held-out images the accuracy is 0.902 and the macro F1 is
0.866. Those are the figures to quote. Section 4.1 of the report has the detail.

## Running it

Open the **folder** (not just the notebook file), then open
`notebooks/brain_tumour_cnn.ipynb` and run all cells. About two minutes.

Nothing needs installing or unzipping first. The first cell installs any missing
packages into whichever kernel is running. The second locates the dataset, and
downloads it from Kaggle if it isn't already on disk.

The dataset is not in this repository — it is 157 MB and not mine to
redistribute. The automatic download needs Kaggle credentials, which take a
minute to set up once: go to kaggle.com/settings, API, *Create New Token*, and
put the downloaded `kaggle.json` in `~/.kaggle/`. If you would rather do it by
hand, download the zip and unpack it so that `data/Training/` and
`data/Testing/` sit next to this README:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

By default the notebook loads the trained weights in `results/` rather than
retraining. Set `RETRAIN = True` in the first code cell to train from scratch —
about two hours on a CPU, ten minutes on a GPU. Seeds are fixed at 42, and a
re-run reproduces the recorded histories exactly.

The MLP baseline weights are not included; that file is 96 MB. Its numbers come
from the saved training history, and `RETRAIN = True` regenerates the weights in
about 11 minutes.

## Files

```
report.pdf       the report (8 pages)
report/          LaTeX source and figures
notebooks/       the notebook, with outputs saved
figures/         every figure as a PNG, with a key to where each one is used
results/         metrics, training histories, model weights, figures
requirements.txt dependencies
```

The notebook is self-contained: preprocessing, the model definitions and the
training loop are all defined inside it, so there is no separate source folder
to keep in step with it.

## Environment

Python 3.11, PyTorch 2.13.0, torchvision 0.28.0, CPU only. Training took 6,980 s
across the three configurations. Seeds are fixed at 42, and a re-run reproduces the recorded histories exactly.
