import argparse
from pathlib import Path

import torch

from model import TinyGPT


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "model.pt"
BLOCK_SIZE = 128


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack(
        [data[start + 1 : start + block_size + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for name, data in (("train", train_data), ("validation", validation_data)):
        batch_losses = []
        for _ in range(10):
            inputs, targets = get_batch(data, batch_size, BLOCK_SIZE, device)
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Expected a loss when targets are provided")
            batch_losses.append(loss.item())
        losses[name] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny GPT-style model.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive integers")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training text not found: {DATA_PATH}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch setup")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    text = DATA_PATH.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    if len(text) <= BLOCK_SIZE + 1:
        raise ValueError(f"{DATA_PATH} must contain more than {BLOCK_SIZE + 1} characters")

    stoi = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([stoi[character] for character in text], dtype=torch.long)
    split = int(0.9 * len(encoded))
    train_data, validation_data = encoded[:split], encoded[split:]
    if len(validation_data) <= BLOCK_SIZE + 1:
        raise ValueError("Training text is too short to create a validation batch")

    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)

    config = {
        "vocab_size": len(vocabulary),
        "block_size": BLOCK_SIZE,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "dropout": 0.1,
    }
    model = TinyGPT(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Device: {device}")
    print(f"Characters: {len(text):,} | Vocabulary: {len(vocabulary)}")
    for step in range(args.steps):
        if step % 100 == 0 or step == args.steps - 1:
            losses = estimate_loss(
                model, train_data, validation_data, args.batch_size, device
            )
            print(
                f"step {step:>4}: train loss {losses['train']:.4f}, "
                f"validation loss {losses['validation']:.4f}"
            )

        inputs, targets = get_batch(
            train_data, args.batch_size, BLOCK_SIZE, device
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Expected a loss when targets are provided")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "config": config,
            "vocabulary": vocabulary,
        },
        args.output,
    )
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(start, max_new_tokens=400)[0].tolist()
    print(f"\nSaved model to: {args.output}")
    print("\n--- Generated sample ---")
    print("".join(vocabulary[index] for index in sample))


if __name__ == "__main__":
    main()
