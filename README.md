# Semi-supervised learning-based neural network with VQ-VAE signatures (SSL-VQ)

`SSL-VQ`  is a machine learning method for predicting therapeutic target molecules for various diseases, leveraging multimodal vector quzntized variation autoencoders (VQ-VAEs).

![alt text](figure1.png)



## Publication/Citation :link:

The study of `SSL-VQ` is described in the manuscript [manuscript](URL). 

```
Satoko Namba, Chen Li, Noriko Yuyama Otani, and Yoshihiro Yamanishi. 
Vector-quantized variational autoencoders for semi-supervised prediction of therapeutic targets across diverse diseases.
Journal name, page number, year. 
```
<br>



## Prerequisites :memo:

The software is developed and tested in Mac OS environment (Apple iMac, Processor 3.6 GHz, 10 core Inter Core i9, Computer memory 128 GB 2267 MHz DDR4, Sonoma 14.4.1).

The basic requirements for running TRESOR are [python>=3.11](https://www.python.org/) with the following packages:
- [python=3.11.6](https://www.python.org/)
- [matplotlib==3.8.2](https://matplotlib.org/)
- [numpy=1.26.2](https://numpy.org/)
- [pandas==2.1.3](https://pandas.pydata.org/)
- [seaborn==0.13.0](https://seaborn.pydata.org/)
- [scikit-learn==1.3.2](https://scikit-learn.org/stable/)
- [torch==2.1.1](https://pytorch.org/)
- [torcheval==0.0.7](https://pytorch.org/torcheval/stable/)

Datails on system requirements is contained in the following file: [requirements.txt](https://github.com/YamanishiLab/SSL-VQ/blob/main/env/requirements.txt).

<br>


## Installation :pick:

Just clone this repository.

```
git clone this-repo-url
cd /path/to/this/repo/
```


## Downloading data :earth_asia:

Please download datasets required for this method from [here](https://yamanishi.cs.i.nagoya-u.ac.jp/sslvq/) or Zenodo [10.5281/zenodo.14644837](10.5281/zenodo.14644837), and put the downloaded `data` folders under the this repository.
```
mv /path/to/the/downloaded/data/PiModel/data/folder /path/to/this/repo/PiModel/
mv /path/to/the/downloaded/data/VAE/data/folder /path/to/this/repo/VAE/
```
<br>



## Contents

- Protein signature construction with VQ-VAE: `/VAE/scr_VQ_VAE_target/`
- Disease signature construction with VQ-VAE: `/VAE/scr_VQ_VAE_disease/`
- Protein signature construction with VAE: `/VAE/scr_VAE_target/`
- Disease signature construction with VAE: `/VAE/scr_VAE_disease/`
- Therapeutic target prediction with SSL-VQ: `/PiModel/`
- Therapeutic target–disease association data of goldstandard set: `/semisupervised/data/`
- Therapeutic target–disease association data of uncharacterized disease set: `/semisupervised/data_old/`
- Requirements: `/env/requirements.txt`



## Usage :technologist:

### 1. Train SSL-VQ model.

Go to the following directory.

```
$ cd ./PiModel/scr_train
```

If you want to train predictive model for inhibitory targets.

```
$ python3 ./01_Train.py \
--fold_number=1 \
--pert_type=trt_sh.cgs \
--gene_hidden_sizes 1024 512 256 \
--gene_epochs=2000
```

If you want to train predictive model for activatory targets.

```
$ python3 ./01_Train.py \
--fold_number=1 \
--pert_type=trt_oe \ 
--gene_hidden_sizes 1024 512 256 \
--gene_epochs=2000
```

The example command parameters mean:

- `--fold_number`: Cross validation fold number (e.g. 1, 2, ..., 5)
- `--pert_type`: Perturbation type of protein perturbation profiles (e.g. trt_sh.cgs or trt_oe)
- `--gene_hidden_sizes`: Hidden layer sizes of neural network (e.g. 1024 512 256)
- `--gene_epochs`: Neural network training epochs (e.g. 2000)



### 2. Predict new therapeutic targets with the trained model.

If you want to predict new inhibitory targets with the trained model.

```
$ python3 ./02_NewPredict.py \
--fold_number=1 \
--pert_type=trt_sh.cgs \
--gene_hidden_sizes 1024 512 256 \
--gene_epochs=2000
```

If you want to predict new activatory targets with the trained model.

```
$ python3 ./02_NewPredict.py \
--fold_number=1 \
--pert_type=trt_oe \ 
--gene_hidden_sizes 1024 512 256 \
--gene_epochs=2000
```

The example command parameters mean:

- `--fold_number`: Cross validation fold number (e.g. 1, 2, ..., 5)
- `--pert_type`: Perturbation type of protein perturbation profiles (e.g. trt_sh.cgs or trt_oe)
- `--gene_hidden_sizes`: Hidden layer sizes of neural network (e.g. 1024 512 256)
- `--gene_epochs`: Neural network training epochs (e.g. 2000)


### Output files

- `saved_gene_nn.pkl`: Trained model
- `output.txt`: Prediction results
- `gene_nn_train_results.csv`: Log data of training process



### 3. Visualization of performance comparisons.

If you want to predict new inhibitory targets with the trained model.

```
$ cd ./performance_evaluation/src
$ python3 ./Visualization_Known_Global.py \
--pert_type=trt_sh.cgs
```

If you want to predict new activatory targets with the trained model.

```
$ cd ./performance_evaluation/src
$ python3 ./Visualization_Known_Global.py \
--pert_type=trt_oe
```

The example command parameters mean:

- `--pert_type`: Perturbation type of protein perturbation profiles (e.g. trt_sh.cgs or trt_oe)

The output figure of performance evaluation of inhibitory target predictions is as follows:

<img src="global.png" width="500">



## License :notebook_with_decorative_cover:

This project is licensed under the LICENSE - see the [LICENSE](https://github.com/YamanishiLab/SSL-VQ/blob/main/LICENCE.txt) file for details



## Contact

For any question, you can contact Yoshihiro Yamanishi ([yamanishi@i.nagoya-u.ac.jp](mailto:yamanishi@i.nagoya-u.ac.jp))
