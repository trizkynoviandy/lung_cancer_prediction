# Lung Cancer Prediction

Lung cancer is the leading cause of cancer death worldwide, accounting for 1.59 million deaths in 2018. The majority of lung cancer cases are attributed to smoking, but exposure to air pollution is also a risk factor. A new study has found that air pollution may be linked to an increased risk of lung cancer, even in nonsmokers.

Predicting lung cancer using machine learning algorithms can help in early diagnosis and treatment, potentially saving lives. In this project, we aims to use XGBoost as a method to classify the severity of cancer on patient.
## Dataset

There are 23 variables used in this project that give us information on on patients with lung cancer, including their age, gender, air pollution exposure, alcohol use, dust allergy, occupational hazards, genetic risk, chronic lung disease, balanced diet, obesity, smoking, passive smoker, chest pain, coughing of blood, fatigue, weight loss, shortness of breath, wheezing, swallowing difficulty, clubbing of finger nails and snoring

Dataset sources: [here](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link).

## Result

The results on the test set are shown below:

| Metric | Score (%) |
| --- | --- |
| Accuracy | 100 |
| Precision | 100 |
| Recall| 100 |
| F1-Score | 100 |

The XGBoost model was able to classify the samples perfectly, achieving an accuracy of 100% on the test set.

Please note that while the model was able to achieve perfect classification on the test set, this does not necessarily imply that the model will generalize well to unseen data. It is always important to validate the model on unseen data and to carefully interpret the results.

## Requirements

In order to run the python script, you will need to have the following packages installed:

* Python
* numpy
* pandas
* scikit-learn

## Disclaimer

This project is for educational purposes only and is not intended to be used for clinical diagnosis or treatment. The results of this model should not be used as a substitute for professional medical advice, diagnosis, or treatment. Please consult a healthcare professional for medical advice. The authors and contributors of this project will not be held responsible for any actions taken based on the results of this model.
