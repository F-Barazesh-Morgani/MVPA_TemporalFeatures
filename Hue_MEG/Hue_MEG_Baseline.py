import os
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import h5py
import concurrent.futures
import multiprocessing

def windowed_data_iteratively(data_hue1, data_hue2, time_bin, window_size):
    data1_windowed = data_hue1[:, :, time_bin]
    data2_windowed = data_hue2[:, :, time_bin]

    # Combine data from both categories
    X = np.concatenate([data1_windowed, data2_windowed], axis=0)  
    y = np.array([1] * data_hue1.shape[0] + [0] * data_hue2.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    return X_scaled, y
       

def train_test_split(X_scaled, y, stimulus_numbers1, stimulus_numbers2, test_stimulus1, test_stimulus2):
    stimulus_numbers = np.concatenate([stimulus_numbers1, stimulus_numbers2])
    train_mask = ~np.isin(stimulus_numbers, [test_stimulus1, test_stimulus2])
    test_mask = np.isin(stimulus_numbers, [test_stimulus1, test_stimulus2])

    X_train = X_scaled[train_mask]  
    Y_train = y[train_mask]  
    X_test = X_scaled[test_mask]    
    Y_test = y[test_mask]    
         
    return X_train, Y_train, X_test, Y_test

def classifier(X_train, Y_train, X_test, Y_test):
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    return accuracy

def evaluate_over_time(data_hue1, data_hue2, time_bins, stim_numbers1, stim_numbers2):
    accuracies = []
    for t in time_bins:
        time_accuracies = []
        X_scaled, y = windowed_data_iteratively(data_hue1, data_hue2, t, window_size = 0)
        for stim1 in np.unique(stim_numbers1):
            for stim2 in np.unique(stim_numbers2):
                if stim1 == stim2:   
                    X_train, Y_train, X_test, Y_test = train_test_split(X_scaled, y, stim_numbers1, stim_numbers2, stim1, stim2)
                    acc = classifier(X_train, Y_train, X_test, Y_test)
                    time_accuracies.append(acc)
        accuracies.append(np.mean(time_accuracies))
    return np.array(accuracies)

# List of participants and files
participants = ["P04", "P16", "P17", "P19", "P22", "P28", "P31", "P33"]
file_map = {
    "P04": "P0004EVC_PCA_200Hz_S1.mat",
    "P16": "P0016EVC_PCA_200Hz_S1.mat",
    "P17": "P0017EVC_PCA_200Hz_S1.mat",
    "P19": "P0019EVC_PCA_200Hz_S1.mat",
    "P22": "P0022EVC_PCA_200Hz_S1.mat",
    "P28": "P0028EVC_PCA_200Hz_S1.mat",
    "P31": "P0031EVC_PCA_200Hz_S1.mat",
    "P33": "P0033EVC_PCA_200Hz_S1.mat"
}

def load_class_dat(data_path, file_name):
    with h5py.File(os.path.join(data_path, file_name), 'r') as f:
        data_group = f['data']
        TrialList = data_group['TrialList'][:].T  
        class_dat = data_group['class_dat'][:].transpose(2, 0, 1) 
    return TrialList, class_dat

def compute_ci(data):
    mean = np.mean(data, axis=0)
    std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])
    ci = 1.96 * std_error
    return mean, ci

def process_participant(participant, data_dir="data/experiment_hue"):
    data_path = os.path.join(data_dir, participant)
    file_name = file_map[participant]

    time_bins = np.arange(0, 221)
    hues = list(range(1, 15))
    hue_data = {}
    mean_accuracies = {}

    for LumOffset in range(1, 4):
        print(f"{participant} LumOffset {LumOffset} started.")
        TrialList, class_dat = load_class_dat(data_path, file_name)
        base_filter = ((TrialList[:, 4] == 1) | (TrialList[:, 4] == 2)) & (TrialList[:, 2] == LumOffset)

        for hue in hues:
            hue_filter = base_filter & (TrialList[:, 1] == hue)
            indices = np.where(hue_filter)[0]
            hue_data[hue] = class_dat[indices]

        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                hue_a = hues[i]
                hue_b = hues[j]
                acc_key = f"{hue_a}v{hue_b}"
                stimulus_numbers_hue_a = np.arange(1, hue_data[hue_a].shape[0] + 1)
                stimulus_numbers_hue_b = np.arange(1, hue_data[hue_b].shape[0] + 1)
                accuracies = evaluate_over_time(hue_data[hue_a], hue_data[hue_b], time_bins, stimulus_numbers_hue_a, stimulus_numbers_hue_b)
                
                if acc_key not in mean_accuracies:
                    mean_accuracies[acc_key] = []
                mean_accuracies[acc_key].append(accuracies)

        print(f"{participant} LumOffset {LumOffset} completed.")

    for pair_key, acc_list in mean_accuracies.items():
        acc_array = np.array(acc_list)
        mean_acc = np.mean(acc_array, axis=0)
        np.savetxt(os.path.join(data_path, f"{participant}_Baseline_mean_{pair_key}.csv"), mean_acc, delimiter=",")
        mean_accuracies[pair_key] = mean_acc

    all_mean_accuracies = np.stack(list(mean_accuracies.values()), axis=0)
    mean_hue_accuracy, ci_hue = compute_ci(all_mean_accuracies)
    np.savetxt(os.path.join(data_path, f"{participant}_Baseline_mean.csv"), mean_hue_accuracy, delimiter=",")
   
    return mean_hue_accuracy


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    Participant_mean_hue_accuracy = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_participant, participants))
        Participant_mean_hue_accuracy.extend(results)

    all_participants_mean_hue_accuracies = np.stack(Participant_mean_hue_accuracy, axis=0)
    overall_mean_hue_accuracy, ci = compute_ci(all_participants_mean_hue_accuracies)

    # Save results to a relative folder
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "Overal_Baseline_mean.csv"), overall_mean_hue_accuracy, delimiter=",")
    np.savetxt(os.path.join(output_dir, "Overal_Baseline_ci.csv"), ci, delimiter=",")

    # Time information
    time = np.arange(-100, 1001, 5)  

    plt.figure(figsize=(10, 6))
    plt.plot(time, overall_mean_hue_accuracy, color='darkcyan', label='Mean Hue Accuracy')
    plt.fill_between(time, overall_mean_hue_accuracy - ci, overall_mean_hue_accuracy + ci, color='darkcyan', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', label='Stimulus Onset')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Chance Level (0.5)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy')
    plt.title('Average Classifier Accuracy Over Time Baseline (w=1ms)')
    plt.legend(loc='lower right')
    plt.xticks(np.arange(-100, 1001, 100))
    plt.yticks(np.arange(0.45, 0.6, 0.05))
    plt.xlim(-100, 1000)
    plt.ylim(0.45, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Overall_Baseline.png"))
    plt.show()
