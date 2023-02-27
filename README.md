# # Widefield_microscopy_denoising

This repository contains three different CNNs for denoising widefield fluorescence microscopy images. The networks are built in Tensorflow 2.7.0 framework.

# Dependencies
```
pip install -r requirements.txt
```

# Notebooks
Notebooks are in the ```notebooks``` folder. 

# Training
```
git clone https://github.com/vebrahimi1990/widefield_microscopy_denoising.git
```

For training, add the directory to your training dataset and a directory to save the model to the configuration file ```(config.py)```.

```
python train.py
``` 


# Evaluation
For evaluation, add the directory to your test dataset and a directory to the saved model to the configuration file ```(config.py)```.

```
python evaluate.py
```


# Architecture
![plot](https://github.com/vebrahimi1990/UNet_RCAN_Denoising/blob/master/image%20files/Architecture.png)

# Results
![plot](https://github.com/vebrahimi1990/UNet_RCAN_Denoising/blob/master/image%20files/Results.png)

# Contact
Should you have any question, please contact vebrahimi1369@gmail.com. 
