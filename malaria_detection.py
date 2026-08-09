import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
DATASET_DIR = "cell_images"          # <-- change if your folder name differs
EPOCHS_1    = 10                 # frozen base training
EPOCHS_2    = 15                 # fine-tuning


# ─────────────────────────────────────────────
#  STEP 1 — DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["Uninfected", "Parasitized"],
    subset="training",
    seed=42
)

val_data = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["Uninfected", "Parasitized"],
    subset="validation",
    seed=42
)

print(f"\nClass mapping: {train_data.class_indices}")
print(f"Training samples:   {train_data.samples}")
print(f"Validation samples: {val_data.samples}\n")


# ─────────────────────────────────────────────
#  STEP 2 — BUILD MODEL (Frozen Base)
# ─────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False   # freeze all base layers initially

# Custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

model.summary()


# ─────────────────────────────────────────────
#  STEP 3 — PHASE 1 TRAINING (Frozen Base)
# ─────────────────────────────────────────────
callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ModelCheckpoint("best_model_phase1.h5", monitor="val_accuracy",
                    save_best_only=True, verbose=1)
]

print("\n===== PHASE 1: Training with frozen base =====\n")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_1,
    callbacks=callbacks_phase1
)


# ─────────────────────────────────────────────
#  STEP 4 — PHASE 2 FINE-TUNING (Unfreeze top layers)
# ─────────────────────────────────────────────
# Unfreeze the last 30 layers of MobileNetV2
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with a much lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    ModelCheckpoint("best_model_finetuned.h5", monitor="val_accuracy",
                    save_best_only=True, verbose=1)
]

print("\n===== PHASE 2: Fine-tuning last 30 layers =====\n")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_2,
    callbacks=callbacks_phase2
)

# Save final model
model.save("malaria_model_final.h5")
print("\nModel saved as malaria_model_final.h5")


# ─────────────────────────────────────────────
#  STEP 5 — PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(h1, h2):
    # Merge both phase histories
    acc     = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss    = h1.history["loss"]         + h2.history["loss"]
    val_loss= h1.history["val_loss"]     + h2.history["val_loss"]
    phase1_end = len(h1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train, val, title in zip(
        axes,
        [acc, loss],
        [val_acc, val_loss],
        ["Accuracy", "Loss"]
    ):
        ax.plot(train, label=f"Train {title}", linewidth=2)
        ax.plot(val,   label=f"Val {title}",   linewidth=2)
        ax.axvline(phase1_end - 1, color="gray", linestyle="--", label="Fine-tune start")
        ax.set_title(f"Model {title}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()

plot_history(history1, history2)


# ─────────────────────────────────────────────
#  STEP 6 — EVALUATION (Confusion Matrix + AUC-ROC)
# ─────────────────────────────────────────────
def evaluate_model(model, val_data):
    val_data.reset()
    y_pred_proba = model.predict(val_data, verbose=1).flatten()
    y_pred       = (y_pred_proba > 0.5).astype(int)
    y_true       = val_data.classes

    print("\n===== Classification Report =====")
    print(classification_report(y_true, y_pred,
                                 target_names=["Parasitized", "Uninfected"]))
    print(f"AUC-ROC Score: {roc_auc_score(y_true, y_pred_proba):.4f}")

    # Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Parasitized", "Uninfected"],
                yticklabels=["Parasitized", "Uninfected"])
    axes[0].set_title("Confusion Matrix", fontsize=14)
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score   = roc_auc_score(y_true, y_pred_proba)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"AUC = {auc_score:.4f}")
    axes[1].plot([0,1], [0,1], color="navy", lw=1, linestyle="--")
    axes[1].set_title("ROC Curve", fontsize=14)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation_plots.png", dpi=150)
    plt.show()

evaluate_model(model, val_data)


# ─────────────────────────────────────────────
#  STEP 7 — GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads       = tape.gradient(loss, conv_outputs)
    pooled_grads= tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap     = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def show_gradcam(img_path, model):
    img       = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    prob  = model.predict(img_array_exp, verbose=0)[0][0]
    label = "Parasitized" if prob > 0.5 else "Uninfected"
    conf  = prob if prob > 0.5 else 1 - prob

    heatmap       = make_gradcam_heatmap(img_array_exp, model)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original        = np.uint8(255 * img_array)
    superimposed    = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, img_show, title in zip(
        axes,
        [original, heatmap_resized, superimposed],
        ["Original", "Grad-CAM Heatmap", f"Overlay\n{label} ({conf:.1%})"]
    ):
        ax.imshow(img_show, cmap="jet" if "Heatmap" in title else None)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.suptitle(f"Prediction: {label} | Confidence: {conf:.1%}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
#  STEP 8 — SINGLE IMAGE PREDICTION
# ─────────────────────────────────────────────
def predict_image(img_path, model):
    img       = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob  = model.predict(img_array, verbose=0)[0][0]
    label = "Uninfected (No Malaria)" if prob > 0.5 else "🦟 Parasitized (Malaria Detected)"
    conf  = prob if prob > 0.5 else 1 - prob

    plt.imshow(img)
    plt.title(f"{label}\nConfidence: {conf:.2%}", fontsize=13,
              color="red" if "Parasitized" in label else "green")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\nResult     : {label}")
    print(f"Confidence : {conf:.2%}")
    return label, float(conf)


# ─────────────────────────────────────────────
#  USAGE EXAMPLES (uncomment when needed)
# ─────────────────────────────────────────────

# show_gradcam("path/to/cell_image.png", model)
# predict_image("path/to/cell_image.png", model)

# To load the saved model later:
# model = tf.keras.models.load_model("malaria_model_final.h5")