# Machine Learning of Raman Spectra Predicts Drug Release from Polysaccharide Coatings for Targeted Colonic Delivery

This repository accompanies the manuscript [*“Machine Learning of Raman Spectra Predicts Drug Release from Polysaccharide Coatings for Targeted Colonic Delivery.”*](https://www.sciencedirect.com/science/article/pii/S0168365924005492) It provides all the code necessary to reproduce the main results, including spectral data processing, model training, cross-validation, model evaluation, and SHAP-based feature importance analysis.

## Repository Overview

```
coating_release/
├── data/
│   └── coating_release.xlsx            # Main dataset
├── checkpoints/
│   └── best_*                          # Saved model files (pickled)
├── new/
│   └── ...                             # Figures, output files, etc.
├── scripts/
│   ├── run_nested_cv.py               # Performs nested cross-validation
│   ├── run_best_models.py             # Validates the best models on an external dataset
│   ├── shapley.py                     # Generates SHAP explanations
│   └── ...                            
├── src/
│   ├── cross_validation.py            # Nested CV logic
│   ├── evaluate_model.py              # Model loading and evaluation
│   ├── process_spectra.py             # Spectral data preprocessing
│   └── ...
├── tests/
│   └── test_*.py                      # Unit tests for various modules
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

- **`data/`** contains the Excel file with Raman spectral data and metadata.  
- **`checkpoints/`** stores the pre-trained pickled models (`.pkl` files).  
- **`new/`** is used for output files (such as `.csv` results, plots, etc.).  
- **`scripts/`** holds command-line scripts you can run to execute analyses or generate figures.  
- **`src/`** contains core Python modules implementing data processing, cross-validation, evaluation, etc.  
- **`tests/`** contains unit tests for validating the code.

## Installation & Dependencies

1. Install [Python 3.11+](https://www.python.org/downloads/) (recommended)  
2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing the Paper’s Results

1. **Run Nested Cross-Validation**  
   ```bash
   python scripts/run_nested_cv.py
   ```  
   This will perform nested cross-validation for each model type, optimising hyperparameters via `RandomizedSearchCV`.  

2. **Validate the Best Models**  
   ```bash
   python scripts/run_best_models.py
   ```  
   This uses the external validation set to estimate final performance metrics of the best-found models.

3. **Additional Scripts**  
   - `scripts/conformal_preds.py`: Trains a random forest with conformal prediction intervals.  
   - `scripts/eda_release.py`: Performs exploratory data analysis on release data.  
   - `scripts/shapley.py`: Generates SHAP plots to interpret model predictions.  

4. **Outputs**  
   After running these scripts, you will find output files (e.g. CSVs, figures) in the `new/` folder, such as:
   - `new/*.csv` for numerical results (scores, predictions, etc.)
   - `new/*.png` or `new/*.pdf` for plots and figures
   - `models/*_best.pkl` for optimised model files

## Contact and Citation

If you use this code or data in your work, please cite the associated manuscript. For queries about implementation details, feel free to [open an issue](#) or contact the authors.