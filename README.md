# LeakGAN-PyTorch
A simple implementation of LeakGAN in PyTorch described in [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624). 


## Requirements
* **PyTorch r1.1.0**
* Python 3.6+
* CUDA 7.5+ (For GPU)


##File
* Discriminator.py: The discriminator model of LeakGAN including Feature Extractor and classification
* data_iter.py: Data loader for Generator and Discriminator
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
Copyright (c) 2019 Nurpeiis Baimukan