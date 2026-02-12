# LeakGAN-PyTorch
A simple implementation of LeakGAN in PyTorch described in [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624).

## Requirements
* **PyTorch 2.0+**
* Python 3.8+
* NumPy
* SciPy
* NLTK (for BLEU evaluation)
* CUDA 11.0+ (For GPU)

## Files
* `Discriminator.py`: The discriminator model of LeakGAN including Feature Extractor and classification
* `Generator.py`: The generator model of LeakGAN including worker and manager units
* `data_iter.py`: Data loader for Generator and Discriminator
* `utils.py`: Contains all the connecting parts for recurrent & loss functions
* `main.py`: Running this file will initiate training
* `train.py`: Alternate training entry point
* `convert.py`: Convert token indices to real words
* `eval_bleu.py`: Evaluation of the BLEU scores (2-5) between test dataset and generated data
* `encode.py`: Text-to-tensor preprocessing and encoding

## Usage
```bash
python main.py
```

## Reference
```bibtex
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
