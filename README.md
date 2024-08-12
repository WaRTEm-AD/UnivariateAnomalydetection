Overview

This is the implementation for Twin-Autoencoder and anomaly detection procedure mentioned in 
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231222011249">"Warping Resilient Robust Anomaly Detection in Time Series"</a>, Neurocomputing Journal, 2023


WARTEm-AD is unsupervised anomaly detection model which efficiently detects all types of anomalies such as point, sequence and sub-sequence anomalies, by learning warp resilent representaion form time series sequences. Anomaly scores of time sequences/points are computed based on these learnt representations using Nearest Neighbour method (K-NN or LOF).

Dataset

Numenta Anomaly Benchmark dataset (NAB)
https://github.com/numenta/NAB

UCR archive time series dataset
https://www.cs.ucr.edu/~eamonn/time_series_data/

Discord dataset
https://www.cs.ucr.edu/~eamonn/discords/

Usage

WaRTEM-AD directory has main code of Twin auto ecoder architecture, Warp operators and representation extraction. This code can be executed easily using code_wartem.ipynb python notebook. Required parameter setting should be provided in dict_to_yaml.py or can be given along with python execution command

For executing the code docker installation has to be done

Docker installation steps (for ubuntu 18.04):
1) sudo apt update
2) sudo apt install apt-transport-https ca-certificates curl software-properties-common
3) sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
4) sudo apt update
5) apt-cache policy docker-ce
6) sudo apt install docker-ce
7) sudo systemctl status docker
8) sudo docker pull nvcr.io/nvidia/tensorflow:19.11-tf1-py3
9) sudo docker run --gpus all -it nvcr.io/nvidia/tensorflow:19.11-tf1-py3 bash
   

After entering docker shell install:

1) pip install keras==2.3.1
2) pip install h5py==2.10.0

to exit from docker:
exit

to list docker containers:
docker ps -a

to stop and remove docker container:
docker stop <container_id>
docker rm <container_id>

to list docker images:
docker images

to remove docker image:
docker rmi <image_id>

to start docker container:
sudo docker start <container_id>

to execute docker:
sudo docker exec -it <container_id> /bin/bash


Data subsequencing for representaion learning and Anomaly scoring after representation learning can be done with point_anomalyscoring.ipynb/seq_anomaly scoring.ipynb
