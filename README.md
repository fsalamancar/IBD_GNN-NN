# Patient Classification using Graph Neural Networks and Protein Interaction Networks


[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![Status](https://img.shields.io/badge/status-Research%20Project-yellow)


## Overview

This repository contains the code and resources for a machine learning project focused on classifying patients (e.g., diseased vs. control) by leveraging **Graph Neural Networks (GNNs)**. Our approach integrates patient-specific protein expression data with a **protein-protein interaction (PPI) network** to learn rich, context-aware representations of proteins and, subsequently, patients. This methodology aims to improve patient stratification, especially in scenarios with imbalanced class distributions.

## 👨‍💻 Authors

- **Francisco Salamanca¹** — MSc Bioinformatics Student, Universidad Nacional de Colombia  
- **Jorge Morales²** — MSc Industrial Engineering Student, Universidad Nacional de Colombia  


## Project Goal

The primary objective is to develop a robust classification model capable of accurately distinguishing between patient groups (e.g., "diseased" and "control") by combining:
1.  **Patient Protein Expression Data:** Quantitative measurements of protein levels in each patient.
2.  **Protein-Protein Interaction (PPI) Network:** A graph representing known functional and physical interactions between proteins, providing crucial biological context.

## Methodology

Our pipeline consists of several key stages:

### 1. Data Preparation and Preprocessing
* **Input Data:**
    * `patient_expression_data`: A matrix ($N_p \times N_{pr}$) where $N_p$ is the number of patients and $N_{pr}$ is the number of proteins.
    * `x_protein_features`: Initial node features for the GNN, where each protein's features are its expression levels across all patients ($N_{pr} \times N_p$).
    * `edge_index`, `edge_weight`: Representing the PPI network's connectivity and interaction strengths.
    * `patient_labels`: Binary labels for each patient.
* **Data Split:** Data is split into training and testing sets.
* **Class Imbalance Handling:** Weighted Cross-Entropy Loss is employed during training to mitigate the effects of class imbalance, assigning different weights to the minority and majority classes.

### 2. Model Architecture: `PatientClassifierGNN`

Our custom `PatientClassifierGNN` model is designed in two main parts:

#### a. Protein Embedding Generation (Graph Neural Network)
A two-layer **Graph Convolutional Network (GCN)** learns context-rich embeddings ($H_{\text{proteins}}$) for each protein.
* **First GCN Layer (`conv1`):** Transforms initial protein features ($H^{(0)}$) by aggregating information from neighboring proteins within the PPI network (represented by its normalized adjacency matrix $\tilde{\mathbf{A}}$) via learnable weights ($\mathbf{W}^{(1)}$). A ReLU activation and Dropout (p=0.5) are applied.
    $$H^{(1)} = \text{ReLU}(\tilde{\mathbf{A}} H^{(0)} \mathbf{W}^{(1)})$$
* **Second GCN Layer (`conv2`):** Further processes intermediate embeddings to yield final protein embeddings ($H_{\text{proteins}}$).
    $$H_{\text{proteins}} = \tilde{\mathbf{A}} H_{\text{intermediate}} \mathbf{W}^{(2)}$$

#### b. Patient Embedding Construction & Classification
* **Patient Embedding Construction:** Unique patient-specific embeddings ($E_{\text{patients}}$) are derived by a matrix multiplication of the original patient expression data ($E_{\text{patient}$) with the learned protein embeddings ($H_{\text{proteins}}$). This integrates network context with personalized expression profiles.
    $$E_{\text{patients}} = \mathbf{E}_{\text{patient}} \times H_{\text{proteins}}$$
* **Patient Classification:** The patient embeddings are then passed through a final linear layer (`fc_patient`) to produce classification logits ($L$).
    $$L = \text{Linear}(E_{\text{patients}})$$

### 3. Training and Evaluation Protocol

* **Loss Function:** Weighted Cross-Entropy Loss for class imbalance.
    $$L(y_{\text{true}}, L_{\text{pred}}) = -\frac{1}{N_p} \sum_{i=1}^{N_p} \sum_{c=0}^{N_{\text{classes}}-1} w_c \cdot \mathbf{1}_{y_{\text{true},i}=c} \cdot \log(\text{softmax}(L_{\text{pred},i})_c)$$
* **Optimizer:** Adam optimizer.
* **Hyperparameter Optimization:** A grid search was performed to find optimal hyperparameters, including learning rates, GNN hidden channels, protein embedding dimensions, and number of epochs.
* **Evaluation Metrics:** Model performance is assessed using:
    * **Test Accuracy**
    * **F1-score (Class 1 - Diseased)**
    * **Precision (Class 1)**
    * **Recall (Class 1)**
    * **F1-score (Class 0 - Control)**
    * **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**
    * A detailed **Classification Report** and **Confusion Matrix** are generated.

## Experimental Results

Our experiments involved comparing the GCN-based model against a Graph Attention Network (GAT) model and a traditional Multi-Layer Perceptron (Classical NN).

### GNN Models (GCN vs. GAT)

| Model | Test Acc | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 0) | AUC  |
| :---- | :------- | :------------------ | :------------------ | :--------------- | :----------------- | :--- |
| **GCN** | 58%      | **0.29** | 0.20                | **0.51** | **0.70** | **0.84** |
| GAT   | 47%      | 0.21                | 0.14                | 0.43             | 0.60               | 0.77 |

**Conclusion:** The **GCN-based model consistently outperformed the GAT-based model** across all metrics, showing a better ability to distinguish between classes (higher AUC) and higher overall accuracy.

### GCN vs. Classical Neural Network

**Note on Test Sets:** The Classical NN was evaluated on a test set with a more severe class imbalance (Class 1 being rarer) compared to the GNN models. This impacts direct accuracy comparison.

| Model             | Test Acc | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 0) |
| :---------------- | :------- | :------------------ | :------------------ | :--------------- | :----------------- |
| **GCN** | 58%      | **0.29** | 0.20                | **0.51** | 0.70               |
| **Classical NN** | **83.14%** | **0.29** | **0.45** | 0.21             | **0.90** |

**Conclusion:**
* Both models achieved an **identical F1-score (0.29) for the minority "Diseased" class**, indicating similar overall balanced performance on this challenging class.
* The **Classical NN** exhibited significantly **higher Precision (0.45) for Class 1**, meaning it produced fewer false alarms (diagnosing healthy patients as diseased). However, its **Recall (0.21) for Class 1 was very low**, implying it missed a large proportion of actual diseased patients.
* The **GCN**, conversely, showed a **higher Recall (0.51) for Class 1**, making it more effective at identifying true diseased cases. This came at the cost of lower Precision (0.20), leading to more false positives.
* While the Classical NN showed a higher overall accuracy due to its strong performance on the overwhelmingly dominant healthy class, the **GCN's higher Recall and AUC (0.84)** suggest its potential value in scenarios where detecting true positive cases is paramount.

## Getting Started

To replicate the results or run the model:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    *(Assuming you use `pip` and a `requirements.txt` file)*
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you have PyTorch and PyTorch Geometric installed, e.g., `torch`, `torch-geometric`, `scikit-learn`, etc.)
3.  **Prepare your data:** Place your `patient_expression_data`, `edge_index`, `edge_weight`, and `patient_labels` in the appropriate directories. The `main.ipynb` notebook likely contains the data loading logic.
4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook main.ipynb
    ```
    Follow the steps in the notebook to train and evaluate the models.

## Repository Structure


├── main.ipynb             # Main Jupyter Notebook containing model definition, training, and evaluation

├── ml_project_paper.pdf   # (Optional) Associated research paper or project report

├── data/                  # Directory for input data (e.g., patient_expression.csv, edges.csv, labels.csv)

│   ├── patient_expression_data.pt # Example: Protein expression tensor

│   ├── edge_index.pt        # Example: Graph edge index tensor

│   └── ...

└── README.md              # This file



## Contributing

We welcome contributions! Please open an issue or submit a pull request for any improvements or bug fixes.


---
## 👤 Author

**Francisco Salamanca**  
Bioinformatician | MSc in Bioinformatics  
Universidad Nacinal de Colombia | Institute of Clinical Molecular Biology
[GitHub](https://github.com/fsalamancar) • [Website](https://fsalamancar.github.io/) • [LinkedIn](https://www.linkedin.com/in/fjosesala/) • [IKMB](https://www.ikmb.uni-kiel.de/people/francisco-salamanca/)
