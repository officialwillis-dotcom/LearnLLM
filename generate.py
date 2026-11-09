import argparse
from pathlib import Path

import torch

from model import TinyGPT
from train import DEFAULT_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained model.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be a positive integer")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run train.py first."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch setup")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    stoi = {character: index for index, character in enumerate(vocabulary)}
    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        parser.error(f"Prompt contains characters not in training data: {unknown!r}")

    model = TinyGPT(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt = args.prompt or vocabulary[0]
    index = torch.tensor([[stoi[character] for character in prompt]], device=device)
    result = model.generate(index, args.tokens)[0].tolist()
    print("".join(vocabulary[token] for token in result))


if __name__ == "__main__":
    main()
