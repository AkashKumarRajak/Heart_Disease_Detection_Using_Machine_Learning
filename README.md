# Heart Disease Detection Using Machine Learning

---

## Project Overview

This project aims to detect the presence of heart disease in patients using a Logistic Regression model from scikit-learn. It leverages 13 clinical features — such as cholesterol, resting blood pressure, chest pain type, and more — to predict the likelihood of heart disease.
---

## 📂 Dataset & Key Features:

The dataset consists of 303 patient records with the following medical features:

Feature	Description:

- age: Age of the patient
- sex: Gender (1 = male; 0 = female)
- cp: Chest pain type (0–3)
- trestbps: Resting blood pressure (mm Hg)
- chol: Serum cholesterol (mg/dl)
- fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- restecg: Resting ECG results (0–2)
- thalach: Maximum heart rate achieved
- exang: Exercise-induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise
- slope: Slope of the ST segment (0–2)
- ca: Number of major vessels colored by fluoroscopy (0–4)
- thal: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
- target: Target variable (1 = disease(Defective), 0 = no disease(Healthy))


---

## Machine Learning Pipeline

- **Data Preprocessing**:
  - Separated features X and target Y
  - Split dataset into training (80%) and testing (20%) sets
  - Scaled features using StandardScaler to improve convergence

- **Model Training**:
 ```python 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)
```

- **Evaluation**:
 ```python 
from sklearn.metrics import accuracy_score

Y_pred = model.predict(X_test_scaled)
score = accuracy_score(Y_test, Y_pred)
print("Accuracy:", score)
```
- Final Accuracy: ~ The accuracy score is 0.8032786885245902

- **Prediction**:
 ```python 
import numpy as np

input_data = (35, 0, 2, 120, 180, 0, 1, 170, 0, 0.0, 2, 0, 1)
input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)

if prediction == 0:
    print("You are Healthy ")
else:
    print("Consult your Doctor ⚠")
```

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn

---

## Conclusion

- This project demonstrates a practical implementation of Logistic Regression to predict heart disease using structured patient data. Despite using a relatively simple model, the system achieves respectable accuracy and offers a foundation for more advanced solutions. By building on this baseline — exploring deeper models and incorporating more real-world data — the project could evolve into a powerful diagnostic support tool for medical practitioners. It highlights how data science can contribute meaningfully to healthcare and early disease detection.

---

## 📬 Contact

**Akash Kumar Rajak**  
📧 Email: [akashkumarrajak200@gmail.com](mailto:akashkumarrajak200@gmail.com)  
💼 GitHub: [AkashKumarRajak](https://github.com/AkashKumarRajak)
🔗 LinkedIn: [AkashKumarRajak](https://www.linkedin.com/in/akash-kumar-rajak-22a98623b/)



