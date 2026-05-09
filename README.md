# DermaScan Pro - AI Diagnostic Suite

**DermaScan Pro** is an advanced AI Diagnostic Suite designed for skin cancer detection. It utilizes an ensemble approach combining Convolutional Neural Networks (CNN), Support Vector Machines (SVM), and Quantum Machine Learning (QML) for high-accuracy clinical analysis and predictions.

## 🌟 Key Features

*   **Ensemble Model Architecture:** Utilizes three distinct machine learning models:
    *   **CNN (ConvNeXt-Tiny):** 91.3% Accuracy
    *   **SVM Processor:** 88.7% Accuracy
    *   **Quantum ML (QML Engine):** 84.2% Accuracy
*   **Clinical Dashboard:** A comprehensive, user-friendly interface that presents diagnostic results, ensemble confidence scores, and probability distributions.
*   **AI Clinical Assistant:** Provides automated summaries and precautionary medical advice based on lesion characteristics.
*   **Robust Diagnostic Categories:** Detects multiple skin conditions including:
    *   Benign (Non-cancerous)
    *   Melanoma (MEL)
    *   Basal Cell Carcinoma (BCC)
    *   Actinic Keratosis / Intraepithelial Carcinoma (AKIEC)
*   **Trained on HAM10000 dataset:** Leverages a balanced version of the prominent HAM10000 dataset.

## 📸 Screenshots

*(To display the screenshots you sent, place the 4 images you have inside a folder named `assets` in your project folder, and name them `image1.jpg`, `image2.jpg`, `image3.jpg`, `image4.jpg`. If you named them differently, update the paths below!)*

![Clinical Dashboard 1](assets/image1.jpg)
![Clinical Dashboard 2](assets/image2.jpg)
![Engine Diagnostics 1](assets/image3.jpg)
![Engine Diagnostics 2](assets/image4.jpg)

## 🛠️ Technology Stack

*   **Backend:** Flask (Python)
*   **Machine Learning:** PyTorch, scikit-learn, timm, joblib
*   **Frontend:** HTML, CSS (Jinja Templates)
*   **Data Processing:** NumPy, Pillow

## 🚀 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/DermaScan-Pro.git
    cd DermaScan-Pro
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model Files Setup**
    Ensure the following model files are placed in the `models/` directory:
    *   `best_convnext_skin_cancer_finetuned.pth`
    *   `hybrid_convnext_svm_model.joblib`
    *   `pca_768_to_12.pkl`
    *   `hybrid_quantum_best.pth`
    
    *(Note: Large model files might be ignored by Git depending on `.gitignore`. If you want to share them, consider using Git LFS or a cloud storage link).*

5.  **Run the Application**
    ```bash
    python app.py
    ```

6.  **Access the Suite**
    Open your browser and navigate to `http://localhost:5000`

## 📊 Models Summary

*   **CNN Model:** Fine-tuned ConvNeXt-Tiny acting as the primary feature extractor and base classifier.
*   **SVM Processor:** Trained on the 768-dimensional features extracted by the CNN, reduced via PCA.
*   **QML Engine:** A 12-qubit Hybrid Quantum model used for visual diagnostic reinforcement.

## 📄 License
This project is for educational and research purposes.
