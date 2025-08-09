# rfh_testdatascience_2

Modelo de predicción.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          
│   ├── Analisis_estacionalidad    <- Notebook for Seasonality Analysis
│   └── Test_Prdo.    <- Notebook for test model and prod
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         rfh_testdatascience_1 and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── rfh_testdatascience_2   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes rfh_testdatascience_1 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── main.py                 <- Model execution
    │
    ├── preprocessing.py        <- Data preprocessing
    │
    └── model.py                <- Classes related to model
```

--------

## Python Version
- Python 3.12.10

---

## Environment Setup
Install all dependencies using:

```bash
pip install -r requirements.txt
```
### Model Execution (`main.py`)

The script `main.py` accepts **3 arguments**:

1. **`group` (str)**  
   Group for which the forecast is to be made (>=0.8M or <0.8M)

2. **`periods` (bool)**  
   Periods we want to predict

3. **`test` (bool)**  
   True if we want to run a test with a data set
   False if we want to directly make the prediction

### 🖥️ Example Commands

#### Train a model with validation data
```bash
python main.py --group >=0.8M --periods 24 --test True
```

#### Predict without evaluation
```bash
python main.py --group >=0.8M --periods 24 --test False
```