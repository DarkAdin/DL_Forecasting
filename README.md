# Deep Learning Forecasting

This repository contains a Python wrapper script that applies three Deep Learning model:

* DeepAR
* N-BEATS
* Temporal Fusion Transformer

and generates predictions and metrics on time-series data.

## An example

As an example, a public dataset containing healthy and pathological ECG waves was used. 10000 healthy samples were concatenated and used in order to generate the predictions and metrics shown below. The file containing the data is ```ecg_normal_filas_10000.csv```.

![](DeepAR_preds.png "DeepAR predictions")
![](NBEATS_preds.png "N-BEATS predictions")
![](TFT_preds.png "TFT predictions")

|MAPE|$R^2$  |MAE |RMSE|MBE  |Pearson|Model |
|----|----|----|----|-----|-------|------|
|0.66|0.94|0.13|0.25|-0.03|0.97   |NBEATS|
|0.99|0.86|0.21|0.39|0.06 |0.93   |TFT   |
|1.37|0.85|0.18|0.39|-0.04|0.93   |DeepAR|


## Usage

The script allows for performing both univariate and multivariate analysis. In the case of the latter, covariates must be specified in their appropriate lists within the script.
