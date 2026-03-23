

# **Solar Power Generation Prediction**

##  Overview  
This project develops a machine learning pipeline to **predict AC Power output** for a solar photovoltaic (PV) plant using high‑resolution operational and weather sensor data. Solar energy production is highly dependent on environmental conditions, making accurate forecasting essential for grid stability, maintenance planning, and economic optimization.

This repository includes the full workflow:  
✔ Data preprocessing  
✔ Exploratory Data Analysis (EDA)  
✔ Model training & hyperparameter tuning  
✔ Deep learning experiments  
✔ Deployment using Flask + Docker  

---

##  Problem Description  
Solar power generation fluctuates with weather conditions such as irradiation, temperature, and cloud cover. These fluctuations create challenges for grid operators who must balance supply and demand in real time.

###  Objective  
Build a predictive model that accurately forecasts **AC Power (kW)** using weather sensor data to help operators:

- Improve grid stability  
- Optimize inverter and module maintenance  
- Enhance economic performance of solar plants  



## Dataset Description  
Data was collected from a solar plant in India over **34 days** at **15‑minute intervals**.

### **1. Generation Data**  
- **Records:** 68,778  
- **Features:**  
  - `DATE_TIME`  
  - `DC_POWER`  
  - `AC_POWER` *(target)*  
  - `DAILY_YIELD`  
  - `TOTAL_YIELD`  

### **2. Weather Sensor Data**  
- **Records:** 3,182  
- **Features:**  
  - `AMBIENT_TEMPERATURE`  
  - `MODULE_TEMPERATURE`  
  - `IRRADIATION`  



##  Exploratory Data Analysis (EDA)

### **AC Power Distribution by Hour**
- **Peak Hours (10–14):** Highest AC output (≈1000–1150 kW)  
- **Low Hours (0–6, 18–23):** Near‑zero output due to lack of sunlight  
- **Transition Hours:** Gradual ramp‑up (7–9) and decline (15–17)  
- **Outliers:** High‑demand or extreme‑temperature days  
- **Variability:** Large IQR during peak hours due to weather fluctuations  

This behavior aligns with typical PV generation patterns in warm climates.

*(Plots included in the repository.)*



##  Modeling and Evaluation

### Models Compared  
1. **Linear Regression** (baseline)  
2. **Random Forest Regressor**  
3. **XGBoost Regressor**  
4. **Deep Neural Networks (Keras/TensorFlow)**  
   - Base DNN  
   - DNN with Dropout  
   - DNN without Dropout  

### Key Findings  
- **Random Forest:** Strong performance; tuning did not improve results.  
- **Base DNN:** R² = 0.985, RMSE = 54.88 kW  
- **DNN with Dropout:** Most unstable; R² = 0.9729  
- **DNN without Dropout:** Best DNN variant; R² = 0.9834, RMSE = 50.69 kW  
- **XGBoost (tuned):**  
  - **Best overall model**  
  - **R² = 0.9865**  
  - Selected as the final model  



##  Technical Stack

| Category | Tools |
|---------|-------|
| Data Engineering | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Deep Learning | `tensorflow` (not required for deployment) |
| Deployment | `Flask`, `Gunicorn`, `Docker` |

---

## How to Run the Project

###  Local Setup (Pipenv)

1. Install Pipenv  
   ```bash
   pip install pipenv
   ```

2. Install dependencies  
   ```bash
   pipenv install
   ```

3. Train the model  
   ```bash
   pipenv run python train.py
   ```

4. Start the Flask prediction service  
   ```bash
   pipenv run python app.py
   ```

> **Note:** TensorFlow is only used in the notebook for DNN experiments and can be excluded from the Pipfile.

5. Open the web interface via `index.html` to input values and generate predictions.



### Docker Deployment

1. Build the Docker image  
   ```bash
   docker build -t solar-prediction .
   ```

2. Run the container  
   ```bash
   docker run -it --rm -p 9696:9696 solar-prediction
   ```

(Or use `docker compose up --build` if configured.)

---

## Repository Structure

```
.
├── app.py                 # Flask API for predictions
├── train.py               # Model training pipeline
├── predict_test.py        # API test client
├── models/                # Saved models (XGB, scaler, etc.)
├── data/                  # Raw and processed datasets
├── templates/             # HTML interface for predictions
├── notebooks/             # EDA + model development notebook
├── figures/               # Plots and visualizations
├── Dockerfile             # Container configuration
├── Pipfile / Pipfile.lock # Dependency management
└── README.md              # Project documentation
```

---

##  Author  
**Beven Mpofu**  
Graduate Data Science Student — Michigan Technological University  
Specializing in ML, predictive modeling, and deployment.

Email:bmpofu@mtu.edu
