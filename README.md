Overview

This is the implementation for Twin-Autoencoder and anomaly detection procedure mentioned in 
<a href=https://www.sciencedirect.com/science/article/abs/pii/S0925231222011249>"Warping Resilient Robust Anomaly Detection in Time Series"</a>, Neurocomputing, 2022


WARTEm-AD is unsupervised anomaly detection model which efficiently detects all types of anomalies such as point, sequence and sub-sequence anomalies, by learning warp resilent representaion form time series sequences. And anomaly scoring time sequences/points with these learnt representations using Nearest Neighbour method.

Dataset

Numenta Anomaly Benchmark dataset (NAB)
https://github.com/numenta/NAB

UCR archive time series dataset
https://www.cs.ucr.edu/~eamonn/time_series_data/

Discord dataset
https://www.cs.ucr.edu/~eamonn/discords/

Usage

WaRTEM-AD directory has main code of Twin auto ecoder architecture, Warp operators and representation extraction. This code can be executed easily using code_wartem.ipynb python notebook. required parameter setting should be provided in dict_to_yaml.py or can be given along with python execution command

Data subsequencing for representaion learning and Anomaly scoring after representation learning can be done with point_anomalyscoring.ipynb/seq_anomaly scoring.ipynb
