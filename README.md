# Chilli Health Classifier

A Streamlit app that uses your Google Teachable Machine Keras model to classify chilli images as **healthy** or **unhealthy**.

## Files

- `app.py` — Streamlit user interface and image preprocessing
- `keras_model.h5` — your trained Teachable Machine model
- `labels.txt` — class labels (`0 healthy`, `1 unhealthy`)
- `requirements.txt` — Python dependencies

## Run locally

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Publish to GitHub

1. Create a new **public** GitHub repository, for example `chilli-health-classifier`.
2. Put `app.py`, `requirements.txt`, `README.md`, `keras_model.h5`, and `labels.txt` in the repository root.
3. Commit and push the files. You can use GitHub's website upload, or run:

```bash
git init
git add app.py requirements.txt README.md keras_model.h5 labels.txt
git commit -m "Add chilli health classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/chilli-health-classifier.git
git push -u origin main
```

## Deploy with Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Choose **Create app** and select your repository, branch `main`, and main file `app.py`.
3. Deploy. Streamlit installs packages from `requirements.txt`.

Keep the model and labels in the same repository folder as `app.py`. The model file is about 2.4 MB, so it is suitable for a normal GitHub repository.

## Model notes

The supplied model declares an input size of `224 × 224 × 3`. The app converts images to RGB, resizes/crops them to 224 × 224, and scales pixel values to `[-1, 1]`, which is the standard Teachable Machine image-model preprocessing. Check a sample prediction against the Teachable Machine preview before relying on results.

This is an image classifier demonstration, not a food-safety certification tool.
