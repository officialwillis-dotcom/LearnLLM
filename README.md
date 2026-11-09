# LearnLLM

A small, from-scratch GPT-style character language model for learning how training works. It uses a tiny decoder-only Transformer and PyTorch; it is an educational project, not a practical ChatGPT replacement.

> A space for tracking progress of, and learning how to build a personal LLM.

## What you will learn

1. **Tokenization:** this project treats each character as a token.
2. **Next-token prediction:** the model sees a sequence and learns to predict the character that follows each position.
3. **Attention:** causal self-attention lets each position use only earlier characters.
4. **Training:** cross-entropy measures prediction error; backpropagation and Adam update the model weights.
5. **Generation:** the trained model predicts one character at a time.

## Requirements

- Windows 10/11
- Python 3.10 or newer
- About 1 GB of free disk space for Python and PyTorch
- CPU works; an NVIDIA GPU can make training faster. A GPU is optional for this deliberately small model.

## Set up in VS Code

1. Install [Python for Windows](https://www.python.org/downloads/windows/) and [Git for Windows](https://git-scm.com/download/win) if they are not already installed. In the Python installer, enable **Add Python to PATH**.
2. In VS Code, open this folder (`LearnLLM`) and open **Terminal > New Terminal**.
3. Create a virtual environment and install PyTorch:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   For an NVIDIA GPU, use the install command recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/) instead of the last command. Select Windows, Pip, Python, and the CUDA option that matches your setup. Then install any other project dependencies listed in `requirements.txt` (currently just PyTorch).

4. Check whether PyTorch can use your GPU:

   ```powershell
   .\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

## Train the model

From the project folder, run:

```powershell
.\.venv\Scripts\python.exe train.py
```

Training reads `data.txt`, prints periodic training and validation loss, saves `runs\model.pt`, and prints a generated sample. The included text is a short, original toy corpus; a tiny dataset makes the model easy to run, but it will mostly learn its training text's style and patterns.

Try a shorter run first, or train longer:

```powershell
.\.venv\Scripts\python.exe train.py --steps 200
.\.venv\Scripts\python.exe train.py --steps 3000 --device cuda
```

The device option accepts `auto`, `cpu`, or `cuda`. `auto` selects CUDA when PyTorch reports it is available. If CUDA was not installed or configured, use `--device cpu`.

Generate text from a saved model:

```powershell
.\.venv\Scripts\python.exe generate.py --prompt "Once "
```

If a prompt contains a character that did not occur in `data.txt`, generation reports an error. Retrain after changing the dataset so its vocabulary includes the prompt.

## Explore and experiment

- Change `n_embd`, `n_head`, `n_layer`, or `block_size` in `train.py` to see how model capacity and memory use change.
- Change `--steps` to compare training time and loss.
- Add your own text to `data.txt`. Only use text you have permission to use; training data affects what the model can reproduce.
- Watch the loss: training loss measures fit on training examples; validation loss checks performance on held-out examples. Lower validation loss is generally a sign of better generalization, but a tiny corpus is not a meaningful benchmark.
- This character-level model has no internet knowledge, instruction tuning, safety training, or broad world knowledge.

## Project files

- `model.py` — the decoder-only Transformer and autoregressive generation.
- `train.py` — data preparation, training loop, evaluation, and checkpoint saving.
- `generate.py` — load a checkpoint and generate text from a prompt.
- `data.txt` — small original sample corpus.
- `requirements.txt` — Python dependencies.

## Create the GitHub repository

On GitHub, create a **public** repository named `LearnLLM` under `officialwillis-dotcom` (or use [the new-repository page](https://github.com/new)). Do not initialize it with a README, license, or `.gitignore`. Then, after installing Git for Windows and signing in to GitHub when prompted, run these commands from this project folder:

```powershell
git init
git add .
git commit -m "Create beginner tiny language model project"
git branch -M main
git remote add origin https://github.com/officialwillis-dotcom/LearnLLM.git
git push -u origin main
```

## Useful VS Code extensions

- **Python** (Microsoft) — Python interpreter selection, run/debug support.
- **Pylance** (Microsoft) — Python IntelliSense, type checking, and navigation.
- **Jupyter** (Microsoft) — useful later for interactive experiments and plots.
- **GitLens — Git supercharged** — makes code history and changes easier to inspect.
- **GitHub Pull Requests** (GitHub) — review and manage GitHub pull requests inside VS Code.
