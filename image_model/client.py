import argparse
import warnings

warnings.filterwarnings("ignore")
from flask_ml.flask_ml_cli import MLCli

from model_server import server as ml_server


def main():
    parser = argparse.ArgumentParser(description="DeepFakeDetector for images")
    cli = MLCli(ml_server, parser)
    cli.run_cli()


if __name__ == "__main__":
    main()
