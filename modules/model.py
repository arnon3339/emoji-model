import tensorflow as tf
from transformers import AutoConfig, AdamW, TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from modules import utils
from sklearn.utils.class_weight import compute_class_weight

# def weighted_binary_crossentropy(class_weights):
def weighted_binary_crossentropy(class_weights_tensor):
    # Convert class weights dictionary to NumPy array
    # class_weights_array = np.array(list(class_weights.values()), dtype=np.float32)
    # class_weights_tensor = tf.constant(class_weights_array, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def loss(y_true, y_pred):
        num_labels = class_weights_tensor.shape[0]  # ✅ Use static shape

        y_true = tf.reshape(y_true, [-1, num_labels])
        y_pred = tf.reshape(y_pred, [-1, num_labels])

        # print("Reshaped y_true shape:", tf.shape(y_true))
        # print("Reshaped y_pred shape:", tf.shape(y_pred))

        # ✅ Compute BCE loss with correct shape
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, axis=-1)  # ✅ Keeps shape (batch_size, num_labels)
        # print("BCE loss raw shape:", tf.shape(bce))  # ✅ Debugging step

        bce = tf.expand_dims(bce, axis=-1)  # ✅ Expands to (batch_size, 1)
        bce = tf.tile(bce, [1, num_labels])  # ✅ Expands to (batch_size, num_labels)
        # print("Fixed BCE loss shape:", tf.shape(bce))  # ✅ Debugging step

        batch_size = tf.shape(y_true)[0]

        # ✅ Ensure class weights shape is correct
        class_weights_tensor_expanded = tf.reshape(class_weights_tensor, [1, -1])
        class_weights_tensor_expanded = tf.tile(class_weights_tensor_expanded, [batch_size, 1])

        # print("Class weights shape:", tf.shape(class_weights_tensor_expanded))

        weighted_bce = tf.multiply(class_weights_tensor_expanded, bce)

        return tf.reduce_mean(weighted_bce)

    return loss

class AiModel:
    def __init__(self, batch_size=16, max_legth=128):
        self.max_legth = max_legth
        self.batch_size =  batch_size

        df = pd.read_csv(CONFIG['path']['dataset'], index_col=False)
        self.train_texts, self.test_texts, self.train_labels, self.test_labels =\
            train_test_split(df.iloc[:, 0], df.iloc[:, 1:], random_state=42, test_size=0.3)
        self.val_texts, self.test_texts, self.val_labels, self.test_labels =\
            train_test_split(self.test_texts, self.test_labels, random_state=42, test_size=0.5)

        self.pretrain_name = "bert-base-uncased"
        
        labels = self.train_labels.values
        self.num_labels = len(labels[0])

        N = labels.shape[0]  # Total samples
        C = labels.shape[1]  # Total classes
        class_counts = np.sum(labels, axis=0)  # Count positive labels per class

        self.class_weights = {i: N / (C * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
        class_weights_array = np.array(list(self.class_weights.values()), dtype=np.float32)
        self.class_weights_tensor = tf.constant(class_weights_array, dtype=tf.float32)

    def get_dataset_for_tensor(self, kind="train"):
        if kind == "test":
            texts = self.test_texts.values.tolist()
            labels = self.test_labels.values.tolist()
        elif kind == "val":
            texts = self.val_texts.values.tolist()
            labels = self.val_labels.values.tolist()
        else:
            texts = self.train_texts.values.tolist()
            labels = self.train_labels.values.tolist()

        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_name)
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=self.max_legth,
                        return_tensors="np")

        class_weights = np.array([self.class_weights[i] for i in range(len(self.class_weights))])

        # Compute sample weights per sample
        sample_weights = np.sum(labels * class_weights, axis=1)

        # Convert labels to TensorFlow tensors
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((dict(tokens), labels_tensor, sample_weights))
        
        # Debug: Print the shape before batching
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # print("Dataset element shapes:", dataset.element_spec)
        
        return dataset


    def get_pretrain(self):
        model = TFAutoModelForSequenceClassification\
          .from_pretrained(self.pretrain_name, num_labels=self.num_labels)
        model.config.problem_type = "multi_label_classification"
        return model

    def fit(self, learning_rate=2e-5):
        model = self.get_pretrain()
        train_dataset = self.get_dataset_for_tensor(kind="train")
        val_dataset = self.get_dataset_for_tensor(kind="val")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=weighted_binary_crossentropy(self.class_weights_tensor),
            metrics=["accuracy", tf.keras.metrics.AUC(multi_label=True)],
            run_eagerly=False, # ✅ Ensures debugging output is accurate
            weighted_metrics=[]
        )


        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=5,
            verbose=1
            )
        
        # ✅ Extract the highest validation AUC
        best_val_auc = max(history.history.get("val_auc", [0]))

        return best_val_auc

    def export(self):
        self.model.save_pretrained("./fine_tuned_bert")
        self.tokenizer.save_pretrained("./fine_tuned_bert")