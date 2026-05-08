# AI Patient Diagnosis System

A Machine Learning powered health diagnosis web application built using **Streamlit** and **Support Vector Machine (SVM)**.  
The system predicts possible diseases based on patient health parameters and symptoms.

---

## 🚀 Features

- 🧠 Disease prediction using Machine Learning
- 📊 Displays Top 3 possible diseases
- ⚠️ Confidence and borderline case warnings
- 🎯 Uses trained SVM model
- 🩺 User-friendly Streamlit interface
- 📈 Input normalization and feature scaling
- 🔍 Debug information for testing

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## 📂 Project Structure

```bash
├── app.py
├── svm_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── features.pkl
└── README.md
```

---

## 📸 Application Preview

The application allows users to:

- Enter patient details
- Select symptoms
- Predict disease
- View confidence scores

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ai-patient-diagnosis.git
cd ai-patient-diagnosis
```

---

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🧪 Input Parameters

The model uses the following features:

| Feature | Description |
|---|---|
| Age | Patient age |
| Gender | Male/Female |
| SysBP | Systolic Blood Pressure |
| DiaBP | Diastolic Blood Pressure |
| Temp | Body Temperature |
| HeartRate | Heart Rate |
| BMI | Body Mass Index |
| Symptoms | Number of selected symptoms |

---

## 🧠 Machine Learning Workflow

1. User enters health details
2. Data converted into DataFrame
3. Features aligned with training data
4. Data scaled using saved scaler
5. SVM model predicts disease
6. Probabilities displayed for top diseases

Code reference available in :contentReference[oaicite:0]{index=0}

---

## 📊 Prediction Output

The system shows:

- ✅ Primary predicted disease
- 📌 Top 3 possible diseases
- ⚠️ Low confidence alerts
- ⚠️ Borderline disease detection

---

## 🔒 Important Note

This project is for:

- Educational purposes
- Research and learning
- ML demonstration

It is **NOT** a replacement for professional medical advice.

---

## 📌 Future Improvements

- Deep Learning integration
- Authentication system
- Database integration

---

## 👨‍💻 Author

Developed by Mohammad Raza, pulkit goyal, shiv bhatia, tushar gandhi

---
## Deploy link
Link : https://visionary-moonbeam-e0d5c9.netlify.app/

## ⭐ Support

If you like this project:

- Star the repository
- Fork the project
- Contribute improvements

---
