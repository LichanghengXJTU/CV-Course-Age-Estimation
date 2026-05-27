"""Age estimation on APPA-REAL — DenseNet121 + ViT-B/16.

Entry point. Subcommands dispatch into src.train and src.eval.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Age estimation on APPA-REAL (DenseNet + ViT)"
    )
    parser.add_argument("--model", choices=["densenet", "vit"], required=True)
    parser.add_argument("--mode", choices=["train", "test", "all"], default="all")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config YAML path")
    parser.add_argument("--data_root", type=str,
                        default="./data/appa-real-release",
                        help="APPA-REAL extracted root directory")
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs in config")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"[main.py stub] model={args.model}, mode={args.mode}, "
          f"data_root={args.data_root}")
    # TODO: wire to src.train / src.eval once those modules exist


if __name__ == "__main__":
    main()
