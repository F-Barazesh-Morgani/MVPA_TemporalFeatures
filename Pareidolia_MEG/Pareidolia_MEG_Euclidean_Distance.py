

import os
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import concurrent.futures
import multiprocessing


# Load data from a file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def windowed_data_iteratively(data1, data2, time_bin, window_size):
    
    num_trials, num_features, num_time_bins = data1.shape

    # Define the window range centered on the time_bin
    half_window = window_size // 2
    start_idx = max(0, time_bin - half_window)  
    end_idx = min(num_time_bins, time_bin + half_window)  

    # Extract data for the specific time bin range
    data1_windowed = data1[:, :, start_idx:end_idx]
    data2_windowed = data2[:, :, start_idx:end_idx] 

    # Combine data from both categories
    X = np.concatenate([data1_windowed, data2_windowed], axis=0)  
    y = np.array([1] * data1.shape[0] + [0] * data2.shape[0])
    
    X_flat = X.reshape(X.shape[0], -1) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    X_reshaped = X_scaled.reshape(X.shape[0],X.shape[1], X.shape[2])  
    X_flattened = X_reshaped.reshape(-1, X.shape[1])  
   
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.99, svd_solver='full')  
    pca.fit(X_flattened) 
    X_pca = pca.transform(X_flattened)   
    X_pca_reshaped = X_pca.reshape(X.shape[0], X_pca.shape[1], X.shape[2])  

    return X_pca_reshaped, y

def train_test_split(X_transformed_pca, y, stimulus_numbers1, stimulus_numbers2, test_stimulus1, test_stimulus2):
    
    stimulus_numbers = np.concatenate([stimulus_numbers1, stimulus_numbers2])

    # Create masks for training and testing based on stimulus numbers
    train_mask = ~np.isin(stimulus_numbers, [test_stimulus1, test_stimulus2])
    test_mask = np.isin(stimulus_numbers, [test_stimulus1, test_stimulus2])

    # Split the data into training and testing sets
    X_train = X_transformed_pca[train_mask]  
    Y_train = y[train_mask]  
    X_test = X_transformed_pca[test_mask]    
    Y_test = y[test_mask]    
            
    return X_train, Y_train, X_test, Y_test
   

def Euclidean_Distance(X):
      
    num_instances = X.shape[0]
    X_transformed = np.zeros((num_instances, num_instances))
    
    for i in range(num_instances):
        for j in range(num_instances):
            X_transformed[i, j] = np.linalg.norm(X[i] - X[j])  
           
    return  X_transformed
         

def classifier_RidgeCV(X_train, Y_train, X_test, Y_test):
     
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))  
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    
    return accuracy 

# Evaluate classifier over time for a stimulus pair
def evaluate_over_time(data1, data2, time_bins, stimulus_numbers1, stimulus_numbers2):
    accuracies = []
    for t in time_bins:
        time_accuracies = []
        X_pca_reshaped, y = windowed_data_iteratively(data1, data2, t, window_size = 20)
        X_transformed = Euclidean_Distance(X_pca_reshaped)
        
        for stim1 in np.unique(stimulus_numbers1):
            for stim2 in np.unique(stimulus_numbers2):
                if stim1 != stim2:  
                
                    X_train, Y_train, X_test, Y_test = train_test_split(X_transformed, y, stimulus_numbers1, stimulus_numbers2, stim1, stim2)
                    accuracy = classifier_RidgeCV(X_train, Y_train, X_test, Y_test)
                    time_accuracies.append(accuracy)
        
        accuracies.append(np.mean(time_accuracies))

    return np.array(accuracies)


# List of participants
participants = ["P01", "P02", "P03", "P07", "P08", "P09", "P10", "P11", "P12"]

# Define time bins
time_bins = np.arange(0, 1101, 5) 

# Windowing parameters
window_size = 20  

# Loop over runs
category1, category2, category3 = 1, 2, 3

# Function to process each participant
def process_participant(participant):
    saving_folder = rf'/data/z5452142/experiment_2/{participant}'  

    # Determine the number of runs
    num_runs = 7 if participant == "P09" else 6
    
    all_accuracies_1v2 = []
    all_accuracies_1v3 = []
    all_accuracies_2v3 = []

    for run in range(1, num_runs + 1):
        
        save_path_1v2 = os.path.join(saving_folder, f'run{run}_cat1v2.pkl')
        save_path_1v3 = os.path.join(saving_folder, f'run{run}_cat1v3.pkl')
        save_path_2v3 = os.path.join(saving_folder, f'run{run}_cat2v3.pkl')

        print(f"Processing {participant} - Run {run}, Category Pair 1v2")
        data_cat1, data_cat2, stimulus_numbers_cat1, stimulus_numbers_cat2 = load_data(save_path_1v2)
        accuracies_1v2 = evaluate_over_time(data_cat1, data_cat2, time_bins, stimulus_numbers_cat1, stimulus_numbers_cat2)
        all_accuracies_1v2.append(accuracies_1v2)

        print(f"Processing {participant} - Run {run}, Category Pair 1v3")
        data_cat1, data_cat3, stimulus_numbers_cat1, stimulus_numbers_cat3 = load_data(save_path_1v3)
        accuracies_1v3 = evaluate_over_time(data_cat1, data_cat3, time_bins, stimulus_numbers_cat1, stimulus_numbers_cat3)
        all_accuracies_1v3.append(accuracies_1v3)

        print(f"Processing {participant} - Run {run}, Category Pair 2v3")
        data_cat2, data_cat3, stimulus_numbers_cat2, stimulus_numbers_cat3 = load_data(save_path_2v3)
        accuracies_2v3 = evaluate_over_time(data_cat2, data_cat3, time_bins, stimulus_numbers_cat2, stimulus_numbers_cat3)
        all_accuracies_2v3.append(accuracies_2v3)

        print(f"{participant} - Run {run} completed.")

    # Convert lists to numpy arrays
    all_accuracies_1v2 = np.array(all_accuracies_1v2)
    all_accuracies_1v3 = np.array(all_accuracies_1v3)
    all_accuracies_2v3 = np.array(all_accuracies_2v3)

    # Save results as CSV
    np.savetxt(f"{participant}_w20_Euclidean_Distance_1v2.csv", all_accuracies_1v2, delimiter=",", header="Accuracy_1v2", comments="")
    np.savetxt(f"{participant}_w20_Euclidean_Distance_1v3.csv", all_accuracies_1v3, delimiter=",", header="Accuracy_1v3", comments="")
    np.savetxt(f"{participant}_w20_Euclidean_Distance_2v3.csv", all_accuracies_2v3, delimiter=",", header="Accuracy_2v3", comments="")
    print(f"{participant} - Files saved successfully.")

    # Compute mean accuracy and confidence intervals
    mean_accuracies_1v2 = np.mean(all_accuracies_1v2, axis=0)
    mean_accuracies_1v3 = np.mean(all_accuracies_1v3, axis=0)
    mean_accuracies_2v3 = np.mean(all_accuracies_2v3, axis=0)

    std_error_1v2 = np.std(all_accuracies_1v2, axis=0) / np.sqrt(all_accuracies_1v2.shape[0])
    confidence_interval_1v2 = 1.96 * std_error_1v2

    std_error_1v3 = np.std(all_accuracies_1v3, axis=0) / np.sqrt(all_accuracies_1v3.shape[0])
    confidence_interval_1v3 = 1.96 * std_error_1v3

    std_error_2v3 = np.std(all_accuracies_2v3, axis=0) / np.sqrt(all_accuracies_2v3.shape[0])
    confidence_interval_2v3 = 1.96 * std_error_2v3

    # Save mean accuracies
    np.savetxt(f"{participant}_w20_Euclidean_Distance_mean_1v2.csv", mean_accuracies_1v2, delimiter=",", header="Mean_Accuracy_1v2", comments="")
    np.savetxt(f"{participant}_w20_Euclidean_Distance_mean_1v3.csv", mean_accuracies_1v3, delimiter=",", header="Mean_Accuracy_1v3", comments="")
    np.savetxt(f"{participant}_w20_Euclidean_Distance_mean_2v3.csv", mean_accuracies_2v3, delimiter=",", header="Mean_Accuracy_2v3", comments="")
    print(f"{participant} - Mean accuracy files saved successfully.")

    # Plot accuracy over time
    time = np.arange(-100, 1001, 5)
    plt.figure(figsize=(10, 6))

    plt.plot(time, mean_accuracies_1v2, color='gray', label='Human Faces vs Pareidolia Objects')
    plt.fill_between(time, mean_accuracies_1v2 - confidence_interval_1v2, mean_accuracies_1v2 + confidence_interval_1v2, color='gray', alpha=0.2)

    plt.plot(time, mean_accuracies_1v3, color='red', label='Human Faces vs Regular Objects')
    plt.fill_between(time, mean_accuracies_1v3 - confidence_interval_1v3, mean_accuracies_1v3 + confidence_interval_1v3, color='red', alpha=0.2)

    plt.plot(time, mean_accuracies_2v3, color='purple', label='Pareidolia Objects vs Regular Objects')
    plt.fill_between(time, mean_accuracies_2v3 - confidence_interval_2v3, mean_accuracies_2v3 + confidence_interval_2v3, color='purple', alpha=0.2)

    plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy')
    plt.title(f'{participant} - Classifier Accuracy Over Time')
    plt.legend()
    plt.xticks(np.arange(-100, 1100, 100))

    plt.yticks(np.arange(0.45, 0.8, 0.05))
    
   
    plt.xlim(-100, 1000)
    plt.ylim(0.45, 0.8)

    plt.savefig(f"{participant}_w20_Euclidean_Distance.png")
    plt.show()
    print(f"{participant} - Plot saved successfully.")
  

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for participant in participants:
            executor.submit(process_participant, participant)






