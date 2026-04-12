import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(left_probs, right_probs, none_probs):
    plt.figure(figsize=(10, 5))
    plt.plot(left_probs,  label="HMM left")
    plt.plot(right_probs, label="HMM right")
    plt.plot(none_probs,  label="HMM none")
    plt.xlabel("Iteration")
    plt.ylabel("Avg log probability")
    plt.title("Baum-Welch convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

def plot_confusion_matrix(matrix):
    classes = ["left", "right", "none"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, matrix[i][j], ha="center", va="center",
                    color="white" if matrix[i][j] > matrix.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()