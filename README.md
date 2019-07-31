# LeakGAN-PyTorch
A simple implementation of LeakGAN in PyTorch described in [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624). 

## Requirements
* **PyTorch r1.1.0**
* Python 3.5+
* CUDA 8.0+ (For GPU)


## File
* Discriminator.py: The discriminator model of LeakGAN including Feature Extractor and classification
* Generator.py: The generator model of LeakGAN including worker and manager units
* data_iter.py: Data loader for Generator and Discriminator
* utils.py: contains all the connecting parts for recurrent & loss functions 
* main.py: running this file will initiate 
* convert.py: Convert one-hot number to real word
* eval_bleu.py: Evaluation of the BLEU scores (2-5) between test dataset and generated data


## Reference
```bash
@article{guo2017long,
  title={Long Text Generation via Adversarial Training with Leaked Information},
  author={Guo, Jiaxian and Lu, Sidi and Cai, Han and Zhang, Weinan and Yu, Yong and Wang, Jun},
  journal={arXiv preprint arXiv:1709.08624},
  year={2017}
}
```
## Acknowledgements
Main source:
1. https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/
2. https://github.com/deep-art-project/Music/blob/master/leak_gan/

Copyright (c) 2019 Nurpeiis Baimukan