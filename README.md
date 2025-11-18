# CMPSC-446-final-project
# Fake News and Misinformation Detection System
## Authors
Huy Nguyen, Alicia Peters, Neville Pochara 
## Roles
- Huy- Explainability & Fact Checking
- Alicia- Data & Preprocessing
- Neville- Modeling & Classification 
## Features
- Uses transformer-based NLP models for classification
- Highlights persuasive linguistic patterns (e.g., fear words, exaggerations)
- Integrates fact-checking APIs with verified datasets
- Includes explainable AI methods
## Datasets
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [PolititFact claims](https://www.politifact.com/)
## Worklflow
1. Collect and clean data
2. Preprocess text using spaCy
3. Apply transformer models (BERT, RoBERTa)
4. Compare against fact-checking references
5. Highlight linguistic and factual inconsi
6. Evaluate accuracy and interpretability
## Technologies
- Python, spaCy
- Fact-checking API integration
- SHAP/LIME for explainability
## Contributions
- Huy
  - first prototype: the folder layout, training pipeline, inference script, preprocessing file, spaCy loader, and sample article collection.
- Alicia
  - full evaluation pipeline (accuracy, precision, recall, F1, confusion matrix) and improved the training setup. Built the confidence bar, enhanced article analysis, and added persuasive-cue highlighting.
- Neville
  - added sentiment-analysis support to the system by updating the spaCy model, integrating VADER, modifying inference to return sentiment scores, updating the CLI app to display them, and adding a requirements file to streamline setup.
