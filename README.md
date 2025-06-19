# Dating App Behavior Analysis

This project explores the factors influencing user success on a fictional dating app. Using a synthetic dataset, we analyze how user profile attributes, app usage patterns, and demographics affect match outcomes. The project includes exploratory data analysis, hypothesis testing, and a predictive model for match success.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Variables](#key-variables)
- [Analysis & Hypotheses](#analysis--hypotheses)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Results](#results)
- [License](#license)

## Project Overview

The goal is to understand the dynamics of online dating by examining:
- User profile attributes (photos, bios, demographics, interests)
- App-specific features (algorithms, design, messaging)
- User behavior (swiping patterns, communication styles)
- Social and psychological factors (biases, societal norms)

## Dataset

- **Source:** [Kaggle: Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset/data)
- **Records:** 50,000
- **Features:** 19 (demographics, app usage, swipes, matches, etc.)

## Project Structure

```
coder_house/
    Dating_app_behavior_coderhouse/
        ProyectoDSAlvaroCancino.ipynb
        ...
    ...
scrapingmaps/
    ...
```

## Key Variables

- `gender`: User’s gender identity
- `sexual_orientation`: User’s sexual orientation
- `location_type`: User’s location type (Urban, Suburban, etc.)
- `app_usage_time_min`: Daily app usage time (minutes)
- `swipe_right_ratio`: Ratio of right swipes to total swipes
- `likes_received`: Number of likes received
- `mutual_matches`: Number of successful matches
- `profile_pics_count`: Number of profile pictures
- `bio_length`: Number of characters in bio
- `message_sent_count`: Number of messages sent
- `match_outcome`: Categorical match outcome

## Analysis & Hypotheses

**H1:** Higher activity leads to more matches  
**H2:** More complete profiles achieve greater success  
**H3:** There are differences based on gender/orientation  
**H4:** Urban users are more successful

The notebook includes:
- Exploratory Data Analysis (EDA)
- Statistical tests (t-test, ANOVA, Chi-square)
- Visualizations (heatmaps, violin plots, boxplots)
- Logistic regression model to predict match success

## How to Run

1. Clone the repository and navigate to the project folder.
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Open [ProyectoDSAlvaroCancino.ipynb](coder_house/Dating_app_behavior_coderhouse/ProyectoDSAlvaroCancino.ipynb) in Jupyter Notebook or VS Code.
4. Run the notebook cells sequentially.

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

Install all dependencies with:
```sh
pip install -r requirements.txt
```

## Results

- **Activity**: Strong positive correlation with mutual matches.
- **Profile completeness**: Longer bios and more photos increase match success.
- **Gender/orientation**: No overall gender effect, but differences exist within some orientations.
- **Location**: No general effect, but some impact for non-binary users.

## Model

The project uses a **logistic regression model** to predict user success (mutual match or date happened) based on profile and behavioral features. The model includes both numerical and categorical variables (such as gender, sexual orientation, and location type) using one-hot encoding and feature scaling.

**Model Highlights:**
- **Features used:** App usage time, messages sent, swipe ratio, profile pictures, bio length, activity score, profile completeness, gender, sexual orientation, and location type.
- **Performance:**  
    - **Accuracy:** 97%
    - **Precision/Recall/F1-score:** All above 0.97 for both classes
    - **ROC-AUC Score:** 0.998 (excellent discrimination between success and non-success)
- **Interpretation:**  
    - The model is highly effective at predicting which users are likely to achieve success on the app, confirming the importance of activity and profile completeness.

You can find the full model implementation and evaluation in the notebook:  
[ProyectoDSAlvaroCancino.ipynb](coder_house/Dating_app_behavior_coderhouse/ProyectoDSAlvaroCancino.ipynb)

## License

This project is for educational purposes. Dataset is synthetic and sourced from Kaggle + additional adjustment from the me, the owner of this project.

---

*For more details, see the full analysis in [ProyectoDSAlvaroCancino.ipynb](coder_house/Dating_app_behavior_coderhouse/ProyectoDSAlvaroCancino.ipynb).*