# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:43:00 2025

@author: z5452142
"""

import os
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import multiprocessing
from scipy.io import loadmat


def windowed_data_iteratively(data_object1, data_object2, time_bin, window_size):
    
    num_trials, num_features, num_time_bins = data_object1.shape

    # Define the window range centered on the time_bin
    half_window = window_size // 2
    start_idx = max(0, time_bin - half_window)  
    end_idx = min(num_time_bins, time_bin + half_window)  
  
    # Extract data for the specific time bin range
    data1_windowed = data_object1[:, :, start_idx:end_idx]
    data2_windowed = data_object2[:, :, start_idx:end_idx]

    data1_reshaped = data1_windowed.reshape(data1_windowed.shape[0], -1)  
    data2_reshaped = data2_windowed.reshape(data2_windowed.shape[0], -1)  
    
    # Combine data from both categories
    X = np.concatenate([data1_reshaped, data2_reshaped], axis=0)   
    y = np.array([1] * data_object1.shape[0] + [0] * data_object2.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.99, svd_solver='full')  
    pca.fit(X_scaled)  
    X_pca = pca.transform(X_scaled)  

    return X_pca, y
       

def train_test_split(X_pca, y, stimulus_numbers1, stimulus_numbers2, test_stimulus1, test_stimulus2):
    # Combine stimulus numbers
    stimulus_numbers = np.concatenate([stimulus_numbers1, stimulus_numbers2])
   
    # Create masks for training and testing based on stimulus numbers
    train_mask = ~np.isin(stimulus_numbers, [test_stimulus1, test_stimulus2])
    test_mask = np.isin(stimulus_numbers, [test_stimulus1, test_stimulus2])

    # Split the data into training and testing sets
    X_train = X_pca[train_mask]  
    Y_train = y[train_mask]  
    X_test = X_pca[test_mask]    
    Y_test = y[test_mask]    
         
    return X_train, Y_train, X_test, Y_test

def classifier(X_train, Y_train, X_test, Y_test):
  
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    
    return accuracy

# Evaluate classifier over time for a stimulus pair
def evaluate_over_time(data_object1, data_object2, time_bins, seq_numbers1, seq_numbers2):
    accuracies = []
    for t in time_bins:
        time_accuracies = []
        X_pca, y = windowed_data_iteratively(data_object1, data_object2, t, window_size = 20)
        
        for stim1 in np.unique(seq_numbers1):
            for stim2 in np.unique(seq_numbers2):
                if stim1 == stim2:   
                    X_train, Y_train, X_test, Y_test = train_test_split(X_pca, y, seq_numbers1, seq_numbers2, stim1, stim2)
                    accuracy = classifier(X_train, Y_train, X_test, Y_test)
                    time_accuracies.append(accuracy)
    
        accuracies.append(np.mean(time_accuracies))
        
    return np.array(accuracies)


# List of participants and corresponding file names
participants = ["P38", "P34", "P45", "P37", "P22", "P30", "P28", "P10", "P33", "P07" ]


file_map = {
    "P38": "sub-38-20_Objects_seq.mat",
    "P34": "sub-34-20_Objects_seq.mat",
    "P45": "sub-45-20_Objects_seq.mat",
    "P37": "sub-37-20_Objects_seq.mat",
    "P22": "sub-22-20_Objects_seq.mat",
    "P30": "sub-30-20_Objects_seq.mat",
    "P28": "sub-28-20_Objects_seq.mat",
    "P10": "sub-10-20_Objects_seq.mat",
    "P33": "sub-33-20_Objects_seq.mat",
    "P07": "sub-07-20_Objects_seq.mat",
}

def load_class_dat(data_path, file_name):
    mat_data = loadmat(os.path.join(data_path, file_name))
    
    data = mat_data['data']
    TrialList = data['TrialList'][0, 0]  
    Class_dat = data['Class_dat'][0, 0]  

    return TrialList, Class_dat

def compute_ci(data):
    mean = np.mean(data, axis=0)
    std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])
    ci = 1.96 * std_error
    return mean, ci

def process_participant(participant, base_dir="data/experiment_object"):
    data_path = os.path.join(base_dir, participant)
    file_name = file_map[participant]

    time_bins = np.arange(0, 701, 4)
    object_labels = list(range(20))

    object_data = {}
    pairwise_accuracies = {}
    sequence_numbers_per_object = {}

    # Windowing parameters
    window_size = 20

    print(f"{participant} started.")

    TrialList, Class_dat = load_class_dat(data_path, file_name)

    # Filter and store data for each object
    for label in object_labels:
        object_filter = (TrialList[:, 1] == label)
        indices = np.where(object_filter)[0]

        selected_indices = indices[12:24]
        object_data[label] = Class_dat[selected_indices]

        seq_numbers = TrialList[selected_indices, 3]
        sequence_numbers_per_object[label] = seq_numbers

    # Evaluate pairwise combinations
    for i in range(len(object_labels)):
        for j in range(i + 1, len(object_labels)):
            object_a = object_labels[i]
            object_b = object_labels[j]

            seq_numbers_object_a = sequence_numbers_per_object[object_a]
            seq_numbers_object_b = sequence_numbers_per_object[object_b]

            acc = evaluate_over_time(
                object_data[object_a],
                object_data[object_b],
                time_bins,
                seq_numbers_object_a,
                seq_numbers_object_b
            )
            key = f"{object_a}v{object_b}"
            pairwise_accuracies[key] = acc

            # Save each pairwise accuracy
            np.savetxt(
                os.path.join(data_path, f"{participant}_w20_Sliding_Window_Flattening_{key}.csv"),
                acc, delimiter=","
            )

    print(f"{participant} completed.")

    # Stack all pairwise accuracies to compute overall mean
    all_accuracies_array = np.stack(list(pairwise_accuracies.values()), axis=0)
    mean_object_accuracy, ci_object = compute_ci(all_accuracies_array)

    # Save overall mean accuracy
    np.savetxt(
        os.path.join(data_path, f"{participant}_w20_Sliding_Window_Flattening_mean.csv"),
        mean_object_accuracy, delimiter=","
    )

    return mean_object_accuracy


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    Participant_mean_object_accuracy = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_participant, participants))
        Participant_mean_object_accuracy.extend(results)

    all_participants_mean_object_accuracies = np.stack(Participant_mean_object_accuracy, axis=0)
    overall_mean_object_accuracy, ci = compute_ci(all_participants_mean_object_accuracies)

    BASE_DIR = "data/experiment_object"
    np.savetxt(
        os.path.join(BASE_DIR, "Overall_w20_Sliding_Window_Flattening_mean.csv"),
        overall_mean_object_accuracy, delimiter=","
    )
    np.savetxt(
        os.path.join(BASE_DIR, "Overall_w20_Sliding_Window_Flattening_ci.csv"),
        ci, delimiter=","
    )

    # Time information 
    time = np.arange(-100, 601, 4) 

    plt.figure(figsize=(10, 6))

    plt.plot(time, overall_mean_object_accuracy, color='darkcyan', label='Mean Object Accuracy')
    plt.fill_between(time, 
                     overall_mean_object_accuracy - ci, 
                     overall_mean_object_accuracy + ci, 
                     color='darkcyan', alpha=0.3)

    # Add stimulus and chance lines
    plt.axvline(x=0, color='black', linestyle='--', label='Stimulus Onset')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Chance Level (0.5)')

    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy')
    plt.title('Average Classifier Accuracy Over Time - Sliding Window Flattening (w=20ms)')
    plt.legend(loc='lower right')

    plt.xticks(np.arange(-100, 601, 100))
    plt.yticks(np.arange(0.45, 0.71, 0.05))
    
    plt.xlim(-100, 600)
    plt.ylim(0.45, 0.7)

    plt.tight_layout()
    plt.savefig("Overall_w20_Sliding_Window_Flattening.png")
    plt.show()
