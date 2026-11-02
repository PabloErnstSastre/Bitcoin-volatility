# Bitcoin Volatility Prediction

This repository provides a streamlined pipeline to:
1. **Download** historical Bitcoin price data (BTC-USD) using `yfinance`.
2. **Calculate** daily returns and annualized volatility.
3. **Fit** a GARCH(1,1) model (using the `arch` library) to forecast future volatility.
4. **Visualize** both historical volatility and forecasted values.

## Quick Start

```bash
# Navigate to the project folder
# Create and activate a virtual environment
python -m venv env
# On Windows (PowerShell):
.\env\Scripts\Activate.ps1
# On Linux/Mac:
# source env/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the pipeline (default: last 3 years of data, 30-day forecast)
python -m src.cli

# Project Structure

bitcoin_volatility/
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── volatility.py
│   ├── model.py
│   ├── visualization.py
│   └── cli.py
└── scripts/
    └── run_all.ps1
```
# CLI Options

python -m src.cli --help

start-date YYYY-MM-DD : Start of the historical period (default: 3 years ago).

end-date YYYY-MM-DD : End of the historical period (default: today).

forecast-days N : Number of days for volatility forecasting (default: 30).

output DIR : Directory to save generated plots (default: outputs).
