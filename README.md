# Gym Churn Lab 🏋️‍♂️📊

**Predict Retention. Visualize Trends. Optimize Revenue.**

Gym Churn Lab is a sophisticated machine learning application designed to help fitness centers predict customer churn, visualize member behavior, and identify at-risk customers. Built with Streamlit and Scikit-Learn, it offers a powerful interface for both individual member analysis and bulk data processing.


## 🚀 Key Features

### 1. 🔮 Prediction Engine
-   **Real-time Risk Assessment**: Input member details (demographics, engagement, behavior) to instantly calculate churn probability.
-   **Actionable Insights**: Classifies members into **Low**, **Medium**, or **High** risk categories.
-   **PDF Reporting**: Generate and download detailed individual risk reports for staff to take action.

### 2. 📊 Analytics Dashboard
-   **Business Intelligence**: Visualize key metrics like Total Customers, Churn Rate, Average Lifetime, and Estimated Revenue.
-   **Interactive Charts**: Explore data distribution through dynamic Plotly charts (Donut charts, Bar graphs, Heatmaps).
-   **Deep Dive**: Analyze the relationship between contract periods, age, additional charges, and retention.

### 3. 📂 Batch Processor
-   **Bulk Analysis**: Upload CSV files containing hundreds of member records.
-   **Automated Scoring**: The system processes the entire batch, appending Churn Predictions and Confidence Scores.
-   **Exportable Results**: Download improved datasets as CSV/JSON for further analysis or CRM integration.


## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd ML
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install streamlit pandas joblib plotly fpdf scikit-learn
    ```

## ▶️ How to Run

Execute the following command in your terminal to launch the application:

```bash
streamlit run streamlit/app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 📁 Project Structure

```
ML/
├── README.md                   # Project Documentation
└── streamlit/                  # Application Source Code
    ├── app.py                  # Main Streamlit Application
    ├── background_magma.jpg    # Background Asset
    ├── final_mlp_pipeline.pkl  # Trained ML Model
    ├── gym_churn_us.csv        # Dataset for Dashboard
    └── ...
```

## 🧠 Model Information

The application utilizes a **Multi-Layer Perceptron (MLP)** pipeline trained on historical gym membership data. It considers features such as:
-   **Demographics**: Gender, Age
-   **Engagement**: Lifetime, Contract Period, Group Visits, Frequency
-   **Behavior**: Additional Charges, Proximity to Gym, Partner Programs


