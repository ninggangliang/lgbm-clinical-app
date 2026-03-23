# 🏥 Clinical Prediction Tool (LGBM)

A Streamlit web application for clinical risk prediction using a pre-trained LightGBM model.

## Features

- **Single patient prediction**: Enter clinical features and get instant risk assessment
- **Batch prediction**: Upload a CSV file for bulk predictions
- **SHAP explanations**: Visual interpretation of each prediction
- **Calibrated probabilities**: Logistic recalibration for reliable probability estimates

## Project Structure

```
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── export_model.py            # Script to export model from training pipeline
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
├── models/
│   ├── FinalModel_LGBM.joblib       # Trained LGBM pipeline
│   ├── Recalibrator_LGBM.joblib     # Logistic recalibrator
│   └── feature_meta.json            # Feature names & descriptions
└── README.md
```

## Setup

### 1. Export model from training pipeline

```bash
# Run your thesis pipeline first, then before cleanup:
python export_model.py
```

### 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

## Disclaimer

This tool is for **research purposes only** and should NOT be used as the sole basis for clinical decision-making.
