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
    parser.add_argument("--validation-set-payload", type=Path, help="Held-out annotation JSON.")
    args = parser.parse_args()

    for path in (args.config, args.training_set_payload):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if args.validation_set_payload is not None and not args.validation_set_payload.is_file():
        parser.error(f"Input file does not exist: {args.validation_set_payload}")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"Cannot create output folder: {error}")

    train_model(args.config, args.training_set_payload, args.output_dir, args.validation_set_payload)


if __name__ == "__main__":
    main()
