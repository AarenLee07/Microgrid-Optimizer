# Microgrid-Optimizer

## About

This repository provides codes for the manuscript submitted to ***Energy and Buildings*** (manuscript number: ENB-D-23-03581, title: _Quantifying the Impact of Building Load Forecasts on Optimizing Energy Storage Systems_, revised and under review).

Smart microgrid energy management requires better load forecast models and better control strategies, within each cluster there are considerable research advances over the past few years. Improving the economic performances in real-world practice asks for seamlessly integration of the two parts, however, comprehensive uncertainty quantification is very limited.  
In this research, we focus on understanding how forecast errors on building electricity load impact economic control performances under model predictive control (MPC) strategies, i.e., quantifying the value of information (VoI) for energy storage management. (The explaination of VoI please refer to the paper once being published.)

We implement a collection of both cutting-edge and common-practice learning algorithms for building load forecast, and formulate a MPC pipeline that uses the forecast information for energy storage control.
Specifically, the simulations based on this repository focus on the inclusion of ***Demand charge*** and load forecast accuracy for economic microgrid MPC. Our findings demonstrate that:

- MPC strategies have heterogeneous sensitivities on forecast errors under different energy pricing schemes. Specifically, they are robust when the electricity bill does not include ***demand charges***, but become extremely sensitive to even tiny errors when demand charge is introduced.

- We uncover that forecasting errors have ***asymmetric*** impact on control performances that underestimations of load consequence a more detrimental effect on MPC performance compared to overestimations. 

The overall workflow:

![image](https://github.com/AarenLee07/Microgrid-Optimizer/blob/main/figures/fig1-workflow.png)

## Contents

This repository includes:

- The raw data we used under `data\UCSD_raw_data`. (A public data set, UCSD campus, is selected for this study, please refer to [this paper](https://aip.scitation.org/doi/10.1063/5.0038650) for more detailed explainations).

- Forecasts of the raw data under `data\load_forecast`, and the trained models are also provided under `load_forecast_models`.

- The MPC model for micro economic management.


A demo notebook is provided as `exp_notebook\demo.ipynb`, you may refer to this file for instructions on using this repository.

## Requirements

***Please note that a `gurobi` license is needed for using this repository.*** Academic free trial license is availabe on [obtain_gurobi_license](https://support.gurobi.com/hc/en-us/articles/12684663118993).



***


_We have to admit that the current code was developed for our own experiment purposes, which may not satisfy the quality of an open simulator to support further development. We are aware of this need and will actively enhance our code to make it benefit more research._

