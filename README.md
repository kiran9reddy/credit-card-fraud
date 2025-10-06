**This project detects fraudulent credit card transactions using machine learning. It includes full support for data preprocessing, model training, and batch predictions using a trained LightGBM model.**

credit-card-fraud/
│
├── data/
│   ├── raw/               # Original dataset
│   ├── processed/         # Scaled / feature-selected data
│   │   ├── X.csv          # Test features
│   │   ├── y.csv          # True labels (optional)
│   │   └── predictions.csv # Model output
│
├── models/
│   ├── fraud_model_lgbm.model  # Trained model
│   └── scaler_lgbm.pkl         # Scaler used during training
│
├── notebooks/
│   ├── train_model.ipynb       # Model training notebook
│   └── predict_model.ipynb     # Notebook for batch predictions
│
├── src/
│   ├── train_model.py          # Script to train model
│   └── predict_model.py        # Script for batch prediction (CLI version)
│
└── README.md


⚙️ Setup Instructions

Clone the repository

git clone https://github.com/yourusername/credit-card-fraud.git
cd credit-card-fraud

Create a virtual environment

python3 -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows


Install dependencies

pip install -r requirements.txt

