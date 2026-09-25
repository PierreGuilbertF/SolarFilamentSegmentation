import argparse
from pathlib import Path
from train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Run a solar filament segmentation experiment."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Model configuration JSON file."
    )
    parser.add_argument(
        "--training-set-payload",
        type=Path,
        required=True,
        help="Training set payload.json file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder.")
    args = parser.parse_args()

    for path in (args.config, args.training_set_payload):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"Cannot create output folder: {error}")

    train_model(args.config, args.training_set_payload, args.output_dir)


if __name__ == "__main__":
    main()
