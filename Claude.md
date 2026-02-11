# Claude.md - LeakGAN-PyTorch

## Project Overview
A PyTorch implementation of LeakGAN (Leaked Adversarial Generator) for text generation, based on the paper [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624).

## Architecture
- **Generator**: Hierarchical model with Manager (goal generation via LSTM) and Worker (token generation via LSTM) components
- **Discriminator**: CNN-based text classifier with multiple filter sizes, highway layer, and dropout
- **Training**: Two-phase approach — pretraining (Generator & Discriminator separately) then adversarial training with rollout-based reward estimation

## Key Files
| File | Purpose |
|------|---------|
| `main.py` | Entry point — orchestrates pretraining and adversarial training |
| `Generator.py` | Generator model (Manager + Worker LSTMs) |
| `Discriminator.py` | CNN discriminator with highway network |
| `utils.py` | Core utilities: recurrent functions, loss functions, reward computation |
| `target_lstm.py` | Target LSTM for evaluation and sampling |
| `data_iter.py` | Dataset classes and DataLoader factories |
| `train.py` | Alternate training entry point (incomplete) |
| `convert.py` | Token index to word conversion |
| `eval_bleu.py` | BLEU score evaluation (2-gram through 5-gram) |
| `encode.py` | Text-to-tensor preprocessing and encoding |

## Configuration
Model and training parameters are stored in JSON files under `params/`:
- `leak_gan_params.json` — Generator and Discriminator architecture parameters
- `train_params.json` — Training hyperparameters (LR, epochs, batch size, etc.)
- `target_params.json` — Target LSTM parameters
- `dis_data_params.json` — Discriminator data loading parameters
- `real_data_params.json` — Real data loading parameters

## Commands
```bash
# Run training
python main.py

# Run with custom arguments
python main.py --batch_size 64 --rounds 150 --g_pretrain_steps 120

# Evaluate BLEU scores
python eval_bleu.py

# Convert generated tokens to words
python convert.py
```

## Dependencies
- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- NLTK (for BLEU evaluation)

## Data Format
- Training corpus stored as `.npy` files (NumPy integer arrays of token indices)
- Vocabulary stored as pickle (`.pkl`) files
- Sequence length: 20 tokens
- Vocabulary size: 5258 tokens
- Padding token: vocab_size (5258), Start token: 0
