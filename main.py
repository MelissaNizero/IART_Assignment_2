import argparse

from src import generate_dataset, train_evaluate, update_presentation, web_app


def build_parser():
    parser = argparse.ArgumentParser(
        description="Employee Burnout Risk Prediction project runner"
    )
    parser.add_argument(
        "command",
        choices=["generate", "train", "presentation", "app"],
        help=(
            "Action to run: generate dataset, train/evaluate models, "
            "update presentation, or start the web app"
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "generate":
        generate_dataset.main()
    elif args.command == "train":
        train_evaluate.main()
    elif args.command == "presentation":
        update_presentation.main()
    elif args.command == "app":
        web_app.main()


if __name__ == "__main__":
    main()
