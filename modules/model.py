import numpy as np
import pandas as pd
import os
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from os import path
from transformers import AutoTokenizer

class AiModel:
    def __init__(self):
        self.__data_path = ("./datasets")
        self.__mapping = pd.read_csv(path.join(self.__data_path, "mapping.txt"),
                                     names=["id", "emoji", "name"], 
                                     index_col=False,
                                     delimiter='\t'
                                     )

    def get_data(self, kind="all"):

        if kind == "train":
            with open(path.join(self.__data_path, "train_text.txt")) as f:
                return {
                        "inputs": [line.strip() for line in f.readlines()],
                        "labels": np.genfromtxt(path.join(self.__data_path, "train_labels.txt"), dtype=np.int8)
                        }
        elif kind == "test":
            with open(path.join(self.__data_path, "test_text.txt")) as f:
                return {
                        "inputs": [line.strip() for line in f.readlines()],
                        "labels": np.genfromtxt(path.join(self.__data_path, "test_labels.txt"), dtype=np.int8)
                        }
        elif kind == "val":
            with open(path.join(self.__data_path, "val_text.txt")) as f:
                return {
                        "inputs": [line.strip() for line in f.readlines()],
                        "labels": np.genfromtxt(path.join(self.__data_path, "val_labels.txt"), dtype=np.int8)
                        }
        else:
            all_files = os.listdir(self.__data_path)
            outputs = []
            for i, j in zip(filter(lambda x: "text" in x, all_files),
                            filter(lambda x: "labels" in x, all_files)):
                with open(path.join(self.__data_path, i)) as f:
                    outputs.append({
                        "inputs": [line.strip() for line in f.readlines()],
                        "labels": np.genfromtxt(path.join(self.__data_path, j), dtype=np.int8)
                        })
            return outputs
    
    def get_dataset_for_tensor(self, kind="all", batch_size=8):
        if kind in ['train', 'test', 'val']:
            train_dataset = self.get_data(kind)
            train_tokens = self.tokenize_inputs(train_dataset['inputs'])
            label_tensors = labels = tf.convert_to_tensor(train_dataset['labels'])
            dataset = tf.data.Dataset.from_tensor_slices((dict(train_tokens), label_tensors))
            return dataset.shuffle(len(train_dataset['inputs'])).batch(batch_size)
        else:
            output = {}
            for k in ['train', 'test', 'val']:
                train_dataset = self.get_data(kind)
                train_tokens = self.tokenize_inputs(train_dataset['inputs'])
                label_tensors = labels = tf.convert_to_tensor(train_dataset['labels'])
                dataset = tf.data.Dataset.from_tensor_slices((dict(train_tokens), label_tensors))
                output[k] = dataset.shuffle(len(train_dataset['inputs'])).batch(batch_size)
            return output
    
    def fit(self, learning_rate=5e-5, epochs=3):
        self.model = self.__get_pretrain()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Use a small learning rate for fine-tuning
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Sparse categorical for integer labels
            metrics=["accuracy"]
        )
        train_dataset = self.get_dataset_for_tensor(kind="train")
        self.model.fit(train_dataset, epochs=epochs)

    
    def tokenize_inputs(self, texts):
        # Load pre-trained BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Tokenize the dataset
        inputs = self.tokenizer(
            texts,
            padding=True,           # Pad to the same length
            truncation=True,        # Truncate long sequences
            max_length=128,         # Maximum length of a sequence
            return_tensors="tf"     # Return TensorFlow tensors
        )

        # Print tokenized input
        # print(inputs)
        return inputs

    def __get_pretrain(self):
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(self.__mapping.index)
        )
        return model

    def show_pretrain(self):
        return self.__get_pretrain()

    def export(self, output=""):
        self.model.save_pretrained("./fine_tuned_bert")
        self.tokenizer.save_pretrained("./fine_tuned_bert")