# Model Analysis and Pipeline Implementation
Compares a fine-tuned large model (Mixtral) with a smaller classifier (BERT) on accuracy, precision, recall, F1, latency, training time and model size, and selects BERT for its higher accuracy.
Builds a reusable Scikit-learn text-classification pipeline (TF-IDF + Logistic Regression) driven entirely by a YAML config file, so settings change without code changes.
The pipeline loads the config, loads the data, builds, trains, evaluates and saves a versioned model, and reaches 94.96% accuracy on a 20 Newsgroups test split.
Includes a compliance check (version, random seed, saved model, accuracy ≥ 0.90) and a short reflection on reproducibility and modularity.
Tech stack: Python, Pandas, Matplotlib, Scikit-learn, PyYAML, Joblib, Jupyter Notebook.
