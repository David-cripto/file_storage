## Download Celeba ##
Download Celeba dataset using the [link](https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM), unzip it and set the variable `path_to_celeba` in training and testing scripts for Celeba accordingly. 
## DSBM-IPMF Celeba-64 ##
To train DSBM for Celeba dataset with IPMF procedure run the following script:
```bash
bash run_celeba_male2female_sergei.sh
```
In this script, you need to change the `path_to_celeba` variable by specifying the path to the Celeba dataset and `path_to_save_info` variable by specifying the path to the directory where you want to save results of the experiment.
## ASBM-IPMF Celeba-64 ##
ASBM-IPMF is learned in two stages, initialization is initially learned, then the D-IPMF procedure takes place.
### Celeba-64, one-sided pretraining ###
To pretrain ASBM for identity male-to-male translation before running D-IPMF procedure run the following script:
```bash
bash train_celeba_64_male2male_ema_T_4_eps_1.sh
```
You should specify the name of the experiment using the `exp_name` variable, this variable will be needed later for initialization in D-IPMF training. Also in this script, you need to change the `data_root` variable by specifying the path to the Celeba dataset. To train D-IPMF iterations, you will also need a second model for female-to-female translation. To do this, you need to change the `dataset` variable to `celeba_female_to_male`, by default this variable is `celeba`
### Celeba-64, D-IPMF ###
To run D-IPMF procedure for Celeba-64 after one-sided pretraining run the following script:
```bash
bash train_celeba_64_T_4_imf_ema_sampling_eps_1.sh
```
In this script, you need to change the `data_root` variable by specifying the path to the Celeba dataset. You also need to initialize models from the pretrain to do this, you need to equate the `exp_forward` and `exp_backward` variable with the name `exp_name` variable from the pretrain respectively for the forward and reverse process. In the `exp_forward_model` and `exp_backward_model` variables, you need to enter the names of the checkpoints from which you want to initialize your models.