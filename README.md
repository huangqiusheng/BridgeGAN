# Bridging-the-Gap-between-Label--and-Reference-based-Synthesis(ICCV 2021)
Tensorflow implementation of [**Bridging the Gap between Label- and Reference-based Synthesis in Multi-attribute Image-to-Image Translation**](https://arxiv.org/pdf/2103.02264).

**Overview architecture**
<p align="center"> <img src="./arch.png" width="70%"><br><center></center></p>

## Experiment Results

- CelebA
<p align="center"> <img src="./l.png" width="95%"><br><center></center></p>
<p align="center"> <img src="./r1.png" width="50%"><br><center></center></p>
<p align="center"> <img src="./r2.png" width="50%"><br><center></center></p>


## Preparation

- **Prerequisites**
    - Tensorflow 1.15
    - Python 2.x with matplotlib, numpy and scipy
- **Dataset**
    - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
	- Images should be placed in ./datasets/
## Quick Start

Exemplar commands are listed here for a quick start.
### dataset
- prepare dataset

    ```console
    python datasets/creat_txt.py --path_MultiPIE 'Path to MultiPIE Dataset' --path_chair 'Path to chair Dataset' --path_300w_LP 'Path to 300w-LP Dataset'
    
    shuf datasets/multiPIE_train_paired.txt -o  datasets/multiPIE_train_paired_shuf.txt
    
    python datasets/creat_tf.py --path_MultiPIE 'Path to MultiPIE Dataset' --path_chair 'Path to chair Dataset' --path_300w_LP 'Path to 300w-LP Dataset'

### Training
- To train with size of 128 X 128

    ```console
    python MultiPIE.py --mode training
    
    python chair.py --mode training
    ```

### Testing
- Example of test

    ```console
    python  MultiPIE.py --mode test --batch_size 1 --model_path 'Path to Training Model'
    
    python  chair.py --mode test --batch_size 1 --model_path 'Path to Training Model'
    ```

## Citation
If this work is useful for your research, please consider citing:
