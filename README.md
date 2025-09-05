
# Using Temporal Features to Improve Accuracy in Multivariate Pattern Analysis (MVPA) of M/EEG Data

This repository contains code to accompany the paper *“Using temporal features to improve accuracy in Multivariate Pattern Analysis (MVPA) of M/EEG data”*.

We evaluate various time series classification methods using three M/EEG datasets:

- **Pareidolia-MEG**  
- **THINGS-EEG**  
- **Hue-MEG**

The repository includes implementations of three baseline methods and three time series classification methods (Euclidean Distance, MiniRocket, and Catch22) with varying temporal windows.

---

## Datasets

### 1. Pareidolia-MEG
The dataset originates from Wardle et al. (2020) and contains MEG recordings (160 sensors) of neural responses to **illusory faces**.

- **Participants:** 21  
- **Stimuli:** 96 visual stimuli (32 illusory faces, 32 matched objects, 32 human faces)  
- **Design:** 6 runs × 384 trials each (2,304 trials total per participant)  

In our analysis, we used MEG responses from 9 participants (with head shape information available) and computed average pairwise classification performance across the three stimulus categories. A ridge regression classifier was employed.

**DOI:** [https://doi.org/10.1038/s41467-020-18325-8](https://doi.org/10.1038/s41467-020-18325-8)

---

### 2. THINGS-EEG
The dataset originates from Grootswagers et al. (2022) and contains EEG recordings (64 channels) in response to a large set of **object images** from the [THINGS stimulus set](https://things-initiative.org/).

- **Participants:** 50  
- **Stimuli:** 22,248 images across 1,854 object concepts  
- **Design:** Each image presented twice; validation data contained 200 images repeated 12 times  

For our analysis, we used EEG responses from 10 participants for 20 object concepts, selecting 200 validation images (10 per concept). We computed pairwise classification performance at the category level using a ridge regression classifier.

**Dataset Access:** [https://openneuro.org/datasets/ds003825/versions/1.2.0](https://openneuro.org/datasets/ds003825/versions/1.2.0)  
**DOI:** [https://doi.org/10.1038/s41597-021-01102-7](https://doi.org/10.1038/s41597-021-01102-7)

---

### 3. Hue-MEG
The dataset originates from Goddard & Mullen (2025) and contains MEG responses (275 sensors) to the **hue space**, designed to examine temporal dynamics of color processing.

- **Participants:** 8  
- **Stimuli:** 14 hues × 3 achromatic offsets (luminance increment, isoluminant, decrement)  
- **Design:** Each condition repeated 84 times across two tasks (color categorization and color discriminability)  

We computed pairwise classification of stimulus hues across both tasks using a ridge regression classifier and a leave-one-trial-out scheme.

**DOI:** [https://doi.org/10.1162/jocn.a.56](https://doi.org/10.1162/jocn.a.56)

---

## Methods

In this study, classifiers were trained to discriminate between:

- **Pareidolia-MEG:** pairwise image categories (human faces vs. pareidolia vs. objects)  
- **THINGS-EEG:** pairwise object categories  
- **Hue-MEG:** pairwise hues  

We evaluated:

- **3 Baseline methods**  
- **3 Time series methods**  
  - Euclidean Distance  
  - MiniRocket  
  - Catch22  

All methods were tested with temporal windows of length *t* ms.

---

## Repository Contents

- `code/` → Python scripts for analyses  
- `README.md` → Project description and dataset links  

This repository provides **code for the three datasets**, implementing **3 baseline** and **3 time series classification methods**.

---

## Citation

If you use this repository, please cite our paper:

> Fatemeh Barazesh, *Using temporal features to improve accuracy in Multivariate Pattern Analysis (MVPA) of M/EEG data*, 2025.

