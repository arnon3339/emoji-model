from modules import model
import tensorflow as tf
import os
import random
import builtins
import toml
with open("config.toml", 'r') as f:
    builtins.CONFIG = toml.load(f)

if __name__ == "__main__":
    model_path = "output/models"
    data_path = "assets/data"

    if not path.exists(model_path):
        os.makedirs(model_path)
    if not path.exists(data_path):
        os.makedirs(data_path)

    parser = argparse.ArgumentParser(
        prog="Drawing digit classifier",
        description="The application use CNN to classify drawing digit.",
    )
    parser.add_argument('-tn', '--tuned', action='store_true', help="use fine-tuned values")
    parser.add_argument('-f', '--fit', action='store_true', help="fit the model")
    parser.add_argument('-t', '--tune', action='store_true', help="fine-tunning the model")
    parser.add_argument('-e', '--export', action='store_true', help="export model")
    parser.add_argument('-mf', '--model-format', default="keras", type=str, help='model format (h5 or onnx)')
    args = parser.parse_args()

    if args.tune:
        tune.run()

    if args.fit:
        if args.tuned:
            model.train.train_model(
                num_filters=CONFIG['tuned']['number_filters'],
                lr=CONFIG['tuned']['learning_rate'],
                dropout_rate=CONFIG['tuned']['dropout_rate'],
                export=args.model_format
            )
        else:
            model.train.train_model(
                num_filters=CONFIG['defalut']['number_filters'],
                lr=CONFIG['defalut']['learning_rate'],
                dropout_rate=CONFIG['defalut']['dropout_rate'],
                export=args.model_format
            )

    elif args.convert_format:
        if path.exists(CONFIG['path']['model']['keras']):
            if args.model_format == "onnx":
                model.train.keras2onnx()
        else:
            if args.tuned:
                model.train.train_model(
                    num_filters=CONFIG['tuned']['number_filters'],
                    lr=CONFIG['tuned']['learning_rate'],
                    dropout_rate=CONFIG['tuned']['dropout_rate'],
                    export=args.model_format
                )
            else:
                model.train.train_model(
                    num_filters=CONFIG['defalut']['number_filters'],
                    lr=CONFIG['defalut']['learning_rate'],
                    dropout_rate=CONFIG['defalut']['dropout_rate'],
                    export=args.model_format
                )
