import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from PIL import Image

# Fungsi untuk membuat ROC Curve
def plot_roc_curve(labels, predictions, class_names):
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 6))

    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(labels == i, predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], label=f"Class {class_name} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()

    return plt

# Fungsi untuk log confusion matrix
def log_confusion_matrix(labels, predictions, class_names, normalize=False, figsize=(8, 6)):
    cm = confusion_matrix(labels, predictions)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    title = 'Confusion Matrix'

    plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=True, fmt=fmt, cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    return plt

# Fungsi untuk menghasilkan gambar classification report
def log_classification_report_as_image(labels, predictions, class_names):
    # Generate classification report
    report_dict = classification_report(labels, predictions, target_names=class_names, output_dict=True)

    # Convert report dict to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(2)

    # Plot the DataFrame as a table
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(report_df.columns))))

    # Save the figure to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    # Convert buffer to PIL image (do not close the buffer)
    return Image.open(buf)  # Return PIL Image directly

# Fungsi untuk log sampel prediksi
def log_sample_predictions(inputs, labels, outputs, class_names, epoch, sample_count=3):
	predictions = torch.argmax(outputs, dim=1)
	sample_images = inputs[:sample_count]

	wandb_images = []
	for i in range(len(sample_images)):
		img = inputs[i].permute(1, 2, 0).cpu().numpy()
		label = class_names[labels[i].item()]
		prediction = class_names[predictions[i].item()]

		wandb_images.append(wandb.Image(img, caption=f"Label: {label}, Pred: {prediction}"))

	# Log the images to Weights & Biases
	wandb.log({"sample_prediction": wandb_images}, step=epoch)