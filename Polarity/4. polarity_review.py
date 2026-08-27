import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from obspy import UTCDateTime, read
import os
from tqdm import tqdm
import pandas as pd
import pickle
# Load the data
import seaborn as sns
import matplotlib.ticker as mtick
from scipy import stats

def load_var(model_name:str):
    model_path = f'./var/{model_name}'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
            print(f"Model '{model_name}' loaded successfully.")
            return loaded_model
    else:
        print(f"Model '{model_name}' does not exist.")

def load_history(model_name:str):
    model_path = f'./model/{model_name}'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
            print(f"Model '{model_name}' loaded successfully.")
            return loaded_model
    else:
        print(f"Model '{model_name}' does not exist.")

# load_model using keras.models.load_modul
        
# Training Data
# X_train = load_var("X_train1.pkl")
X_train = load_var("X_train2.pkl")
Y_train = load_var("Y_train.pkl")
print(X_train.shape, Y_train.shape)
print(X_train.dtype, Y_train.dtype)

# Validation Data
# X_validation = load_var("X_validation1.pkl")
X_validation = load_var("X_validation2.pkl")
Y_validation = load_var("Y_validation.pkl")
print(X_validation.shape, Y_validation.shape)
print(X_validation.dtype, Y_validation.dtype)

# Test Data
# X_test = load_var("X_test1.pkl")
X_test = load_var("X_test2.pkl")
Y_test = load_var("Y_test.pkl")
print(X_test.shape, Y_test.shape)
print(X_test.dtype, Y_test.dtype)


print([np.isnan(X_train).any(), 
       np.isnan(Y_train).any(), 
       np.isnan(X_validation).any(), 
       np.isnan(Y_validation).any(), 
       np.isnan(X_test).any(), 
       np.isnan(Y_test).any()]
      )  # Check for NaNs in all variables
print([np.isinf(X_train).any(), 
       np.isinf(Y_train).any(), 
       np.isinf(X_validation).any(), 
       np.isinf(Y_validation).any(), 
       np.isinf(X_test).any(), 
       np.isinf(Y_test).any()]
      )  # Check for infinite values in all variables

    
def probabilityToPolarity(probability_value, threshold=0.5):
    """
    Convert predicted probability values to polarities (0 or 1).
    """
    # Convert each probability value to the corresponding class
    polarities = np.argmax(probability_value, axis=1)  # Assuming classes are represented as 0, 1, 2
    return polarities

def getConfusionMatriks(prediction_label, true_label):
    """
    Melakukan perhitungan performa hasil prediksi dengan nilai yang sebenarnya.\n
    Mengembalikan nilai sebagai berikut:\n
    accuracy, precision, recall, f1_score, confusion_matrix
    """
    TP = FP = TN = FN = 0
    prediction_label = probabilityToPolarity(prediction_label)
    for true_label, pred_label in zip(true_label, prediction_label):
        # print(true_label, pred_label)
        if true_label == 1 and pred_label == 1:
            TP += 1  # True Positive
        elif true_label == 0 and pred_label == 1:
            FP += 1  # False Positive
        elif true_label == 0 and pred_label == 0:
            TN += 1  # True Negative
        elif true_label == 1 and pred_label == 0:
            FN += 1  # False Negative
    # Calculate Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Calculate Precision (handling division by zero)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # Calculate Recall (handling division by zero)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # Calculate F1-Score (handling division by zero)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    # confusion_matrix = np.array([[TN, FP], [FN, TP]])
    confusion_matrix = np.array([[TP/(TP+FN), FN/(TP+FN)], [FP/(TN+FP), TN/(FP+TN)]])
    print(confusion_matrix)
    return(accuracy,precision,recall,f1_score,confusion_matrix)


def plot_history(history, title =""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(range(len(history['loss'])),history['loss'], c="blue", label="Loss")
    axs[0].plot(range(len(history['val_loss'])),history['val_loss'], c="blue",linestyle="--", label="Val Loss")
    axs[0].legend()
    axs[1].plot(range(len(history['accuracy'])),history['accuracy'], c="red", label="Accuracy")
    axs[1].plot(range(len(history['val_accuracy'])),history['val_accuracy'], c="red",linestyle="--", label="Val-Accuracy")
    axs[1].legend()
    fig.suptitle(f"{title}_History")
    fig.savefig(f"{title}_History.png")
    # plt.show()
    plt.close()
    
def plotConfusionMatrix(accuracy,precision,recall,f1_score,confusion_matrix,title=" "):
    """
    Plot hasil performa model berdasarkan hasil perhitungan matrix
    """
    plt.rcParams.update({'font.size': 15})  # Increase global font size (adjust as needed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot confusion matrix heatmap in the first subplot
    sns.heatmap(confusion_matrix, annot=True, fmt=".4f", cmap="Blues", 
                xticklabels=['Predicted Up', 'Predicted Down'],
                yticklabels=['Actual Up', 'Actual Down'], ax=ax1)
    ax1.set_ylabel('Actual Label')
    ax1.set_xlabel('Predicted Label')
    ax1.set_title(f'Confusion Matrix {title}')

    # Plot performance in bar graph in the 2nd subplot
    metrics = [accuracy, precision, recall, f1_score]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    combined_labels = [f'{name}\n({value*100:.2f}%)' for name, value in zip(metrics_names, metrics)]    
    bar_plot = sns.barplot(x=combined_labels, y=metrics, ax=ax2, palette='pastel', hue=combined_labels, legend=False)

    # Automatically add labels to the bars
    for bar in bar_plot.patches:
        height = bar.get_height()
        label = f'{height*100:.2f}%'  # Display the value above the bar
        ax2.annotate(label, 
                    (bar.get_x() + bar.get_width() / 2, height), 
                    ha='center', va='bottom', 
                    fontsize=15)  # Larger font size for bar labels

    ax2.set_ylabel('Score', fontsize=16)  # Larger font size for y-label
    ax2.set_title(f'Performance {title}', fontsize=18)  # Larger font size for title

    # Set y-ticks to display only 0%, 20%, 40%, 60%, 80%, 100%
    yticks = np.arange(0, 1.25, 0.2)  # Create an array for y-ticks
    ax2.set_yticks(yticks)  # Set the y-ticks
    ax2.set_ylim(0, 1.08)  # Set y-axis limit from 0 to 1
    ax2.set_yticklabels([f'{int(tick * 100)}%' for tick in yticks], fontsize=15)  # Set labels for y-ticks
    # ax2.border
    ax2.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()

    # Save and Display the plot
    fig.savefig(f"{title}_Matrix.png")
    # plt.show()
    plt.close()
    
import glob

model_files = glob.glob("model\\selection\\model2*.h5")
print(model_files)

import csv

confusion_matrix_csv = 'confusion_matrix_results--Train Data2.csv'
history_csv = 'model_history_results--Train Data2.csv'



with open(confusion_matrix_csv, 'w', newline='') as cm_file:
    # Define CSV writers
    cm_writer = csv.writer(cm_file)
    # Write headers for confusion matrix CSV
    cm_writer.writerow(["Model ID", "Accuracy", "Precision", "Recall", "F1-Score", "Confusion Matrix"])
    
    for model_file in model_files:
        model = models.load_model(model_file)
        # id = model_file.replace('model\\model_','').replace(".h5","")
        id = model_file.replace('model\\selection\\model2_','').replace(".h5","")
        prediction = model.predict(X_train)
        accuracy,precision,recall,f1_score,confusion_matrix = getConfusionMatriks(prediction,Y_train)
        # prediction = model.predict(X_validation)
        # accuracy,precision,recall,f1_score,confusion_matrix = getConfusionMatriks(prediction,Y_validation)
        # prediction = model.predict(X_test)
        # accuracy,precision,recall,f1_score,confusion_matrix = getConfusionMatriks(prediction,Y_test)
        plotConfusionMatrix(accuracy,precision,recall,f1_score,confusion_matrix,title=f"2_{id}-Train Data")
        cm_writer.writerow([f"2_{id}-Train Data", accuracy, precision, recall, f1_score, confusion_matrix])
    