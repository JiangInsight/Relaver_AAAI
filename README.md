# Relaver: Resolving Latency and Inventory Risk in Market Making with Reinforcement Learning
This repo provides the code for reproducing the stock trading experiments in the IJCAI'25 submission **Resolving Latency and Inventory Risk in Market Making with Reinforcement Learning**. 

 Overview of the Ralaver framework

![overview](overview.png)



### Dependencies
```
Python 3.7 
tensorflow 2.11.0
gym 0.21.0
gym-minigrid 1.0.2
numpy 1.19.5
yfinance 0.2.3
finrl 0.3.5 
```

### Usage
We provide the RElaver implementation on four major Chinese stock index option: IC (CSI 500), IF (CSI 300), IH (SSE 50), and IM (CSI 1000). 
To execute the training and evaluation, specify the ``<index optio_name>`` (``IC`` or ``IF``) first and execute st_run.sh:

```shell
st_run.sh --market <market_name>
```
The testing results on four major Chinese stock index option and the corresponding case studies are as follows:

![overview](result.png)
![overview](result2.png)


