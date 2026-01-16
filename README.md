___________________ MUSIC GENRE CLASSIFICATION PROJECT ~ README ___________________

"Multi-Genre Music Classification Using Hybrid CNN-LSTM Architecture on Raw Audio Waveforms"

Traditional music genre classification systems rely heavily on handcrafted features like MFCCs, spectral contrast, and chroma features. 
This project explores an end-to-end deep learning approach that processes raw audio waveforms directly, eliminating the need for manual feature engineering. 
By combining 1D CNNs (for local temporal patterns) with LSTMs (for sequential dependencies), the model learns hierarchical representations directly from time-domain audio signals, potentially capturing subtle genre characteristics that engineered features might miss.

___________________ PROJECT SETUP INSTRUCTIONS ___________________

1. FILE ARRANGEMENT:

C:/Users/[YourUsername]/Documents/UE/ML/Project/
│
├── Data/                             # REQUIRED: Download from Kaggle
│   ├── genres_original/              # 10 folders with 100 WAV files each
│   │   ├── blues/
│   │   ├── classical/
│   │   ├── ... (10 genres)
│   │
│   ├── features_30_sec.csv          # Pre-extracted features
│   └── features_3_sec.csv
│
├── music_genre_classification.ipynb  # MAIN NOTEBOOK (this file)
├── requirements.txt                  # Python dependencies
│
├── processed_data/                   # AUTO-GENERATED (after first run)
│   ├── X_audio_train.npy
│   ├── y_audio_train.npy
│   └── ... (other .npy files)
│
├── saved_models/                     # AUTO-GENERATED
│   ├── best_cnn_model.h5
│   ├── best_lstm_model.h5
│   ├── best_minimal_hybrid.h5
│   └── best_baseline_model.pkl
│
└── results/                          # AUTO-GENERATED
    ├── baseline_confusion_matrix.png
    ├── cnn_confusion_matrix.png
    ├── hybrid_confusion_matrix.png
    └── final_model_comparison.png

2. DATASET DOWNLOAD INSTRUCTIONS:

> Go to this Kaggle link:: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
> Download the dataset
> Extract to:: C:/Users/[YourUsername]/Documents/UE/ML/Project/Data/
> Verify structure:: Data/genres_original/ should contain 10 folders (blues, classical, ...)

3. ENVIRONMENT SETUP:

> Run these commands in order:

# 1. Install Python 3.8+ if not installed
# 2. Open Command Prompt / Terminal
# 3. Navigate to project folder:
cd "C:/Users/[YourUsername]/Documents/UE/ML/Project"

# 4. Create and activate virtual environment (optional but recommended):
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux

# 5. Install all dependencies:
pip install -r requirements.txt

> If requirements.txt is missing, install the dependencies manually by running the below code:
pip install numpy==1.24.3 pandas==1.5.3 matplotlib==3.7.1 seaborn==0.12.2 scikit-learn==1.3.0 librosa==0.10.0 tensorflow==2.13.0 jupyter==1.0.0 soundfile==0.12.1

4. EXECUTION INSTRUCTIONS:

>> METHOD A: Run complete notebook (RECOMMENDED)
(1) Open Jupyter Notebook: jupyter notebook music_genre_classification.ipynb
(2) Run all cells in order:
    - Click Kernel > Restart & Run All
    - OR press Shift + Enter through each cell
(3) Expected Runtime: 20 - 40 minutes (depending on your system)

>> METHOD B: Run Specific Sections
> Run cells in EXACTLY this order:
(1) Cell 1-2: Import libraries and setup
(2) Cell 3: Load and verify dataset
(3) Cell 4-5: Exploratory Data Analysis (EDA)
(4) Cell 6: Data Preprocessing
(5) Cell 7: Baseline Models
(6) Cell 8: CNN model
(7) Cell 9: LSTM model
(8) Cell 10: Hybrid CNN-LSTM model
(9) Cell 11: Final model comparison and results

5. TROUBLESHOOTING COMMON ERRORS:
(1) Error: "File not found" or dataset path error
    Solution: Update the path in Cell 3

        # CHANGE THIS LINE:
        data_path = "C:/Users/alffa/Documents/UE/ML/Project/Data/genres_original"
        # TO YOUR PATH:
        data_path = "C:/Users/[YourUsername]/Documents/UE/ML/Project/Data/genres_original"

(2) Error: No module named "librosa", or other import errors:
    Solution: Install the missing packages

        pip install librosa tensorflow scikit-learn

(3) Error: Memory error during model training
    Solution: Reduce batch size (Cell 8, 9, 10)

        # CHANGE from 32 to 16 or 8:
        batch_size=16  # or batch_size=8

(4) Error: "Negative Dimension Size" in CNN/LSTM
    Solution: Already fixed in code. If occurs, reduce strides

        # In model definitions, change:
        strides=128 → strides=64
        strides=64 → strides=32

6. VERIFYING SUCCESSFUL EXECUTION:
>> Check if these files are generated:

(1) Models saved:
    > best_cnn_model.h5
    > best_lstm_model.h5
    > best_minimal_hybrid.h5

(2) Visualisations created: 
    > baseline_confusion_matrix.png
    > cnn_confusion_matrix.png
    > lstm_confusion_matrix.png
    > hybrid_confusion_matrix.png
    > final_model_comparison.png

(3) Console output shows:
    > 'Training completed!' for each model
    > Accuracy scores for all models
    > Classification reports

7. QUICK TEST (2 MINUTES):
>> Run this quick test below, to verify setup:

    # In a new cell, run:
    import librosa
    import tensorflow as tf
    import numpy as np
    import pandas as pd

    print("TensorFlow version:", tf.__version__)
    print("Librosa version:", librosa.__version__)

    # Test audio loading
    test_path = "./Data/genres_original/blues/blues.00000.wav"
    audio, sr = librosa.load(test_path, sr=22050, duration=3)
    print(f"Audio loaded: {len(audio)} samples at {sr}Hz")

    print("*** Setup verified successfully!")

8. CONTACT & SUPPORT:
>> If the code does not run:
(1) Check all file paths match your directory structure (given above)
(2) Ensure all dependencies are installed.
(3) Run the "Quick Test" above.
(4) Reduce the dataset size for testing: 
    Change 'samples_per_genre = 20' to 'samples_per_genre = 5'
(!) For further queries, feel free to reach out to our team at ali.farooq@ue-germany.de or vidya.ponnala@ue-germany.de 
