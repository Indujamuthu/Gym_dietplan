#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "age": [25, 45, 60, 30, 50, 40],
    "weight": [65, 80, 75, 70, 90, 85],
    "height": [170, 165, 160, 175, 180, 155],
    "heart_stroke": [0, 1, 0, 0, 1, 0],
    "diabetes": [0, 1, 1, 0, 1, 0],
    "asthma": [1, 0, 1, 0, 0, 1],
    "bmi": [22.5, 29.4, 29.3, 22.9, 27.8, 35.4],
    "diet_plan": ["Low Carb", "Balanced", "Diabetic-Friendly", "High Protein", "Heart-Healthy", "Vegan"]
}

df = pd.DataFrame(data)

# Convert diet_plan into numeric labels
label_map = {label: idx for idx, label in enumerate(df["diet_plan"].unique())}
df["diet_plan"] = df["diet_plan"].map(label_map)

# Split dataset
X = df.drop(columns=["diet_plan"])
y = df["diet_plan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model **with label_map**
with open("diet_model.pkl", "wb") as file:
    pickle.dump({"model": model, "label_map": label_map}, file)

print("Model and label map saved successfully as 'diet_model.pkl'.")


# In[11]:


import gradio as gr
import numpy as np
import pickle

# Load model
with open("diet_model.pkl", "rb") as file:
    data = pickle.load(file)
    model = data["model"]
    label_map = data["label_map"]

# Reverse mapping for diet plan names
reverse_label_map = {v: k for k, v in label_map.items()}

# Define diet recommendations
diet_recommendations = {
    "Low Carb": {"Recommended": ["Lean meats", "Eggs"], "Restricted": ["Sugary foods", "Bread"]},
    "Balanced": {"Recommended": ["Fruits", "Vegetables"], "Restricted": ["Processed foods"]},
    "Diabetic-Friendly": {"Recommended": ["Leafy greens", "Nuts"], "Restricted": ["High sugar fruits"]},
    "High Protein": {"Recommended": ["Chicken", "Fish"], "Restricted": ["Junk food"]},
    "Heart-Healthy": {"Recommended": ["Oats", "Salmon"], "Restricted": ["Red meat"]},
    "Vegan": {"Recommended": ["Legumes", "Fruits"], "Restricted": ["Dairy", "Meat"]}
}

# Prediction function
def recommend_diet(age, weight, height, heart_stroke, diabetes, asthma):
    heart_stroke = 1 if heart_stroke == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    asthma = 1 if asthma == "Yes" else 0
    bmi = weight / ((height / 100) ** 2)
    
    input_data = np.array([[age, weight, height, heart_stroke, diabetes, asthma, bmi]])
    diet_idx = model.predict(input_data)[0]
    diet_plan = reverse_label_map[diet_idx]

    return {
        "Diet_Plan": diet_plan,
        "Recommended": diet_recommendations[diet_plan]["Recommended"],
        "Restricted": diet_recommendations[diet_plan]["Restricted"]
    }

# Gradio Interface
iface = gr.Interface(
    fn=recommend_diet,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Weight (kg)"),
        gr.Number(label="Height (cm)"),
        gr.Radio(["Yes", "No"], label="Heart Stroke"),
        gr.Radio(["Yes", "No"], label="Diabetes"),
        gr.Radio(["Yes", "No"], label="Asthma")
    ],
    outputs="json",
    title="Diet Recommendation System",
    description="Enter your details to get a personalized diet recommendation."
)

iface.launch(share=True)


# In[ ]:




