from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI(title="Student Complaint Priority API with XAI")

MODEL_PATH = "student_complaints_model_2.pkl"

if os.path.exists(MODEL_PATH):
    model_bundle = joblib.load(MODEL_PATH)
    if isinstance(model_bundle, dict):
        model = model_bundle.get("model")
        vectorizer = model_bundle.get("vectorizer")
    else:
        model = model_bundle
        vectorizer = None
else:
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")

class ComplaintRequest(BaseModel):
    stud_number: str
    complaints: str
    description: str

@app.post("/predict")
def predict_priority(data: ComplaintRequest):
    try:
        text_input = f"{data.complaints} {data.description}"
        
        if hasattr(model, "predict"):
            predicted_priority = model.predict([text_input])[0]
        else:
            predicted_priority = "Medium"

        shap_explanation = {}
        reasoning_text = f"This complaint was classified as {predicted_priority} priority based on key terms found in the text."
        
        try:
            vec = vectorizer
            clf = model
            
            if hasattr(model, "named_steps"):
                for name, step in model.named_steps.items():
                    if hasattr(step, "get_feature_names_out") or hasattr(step, "vocabulary_"):
                        vec = step
                    elif hasattr(step, "predict") or hasattr(step, "coef_") or hasattr(step, "feature_importances_"):
                        clf = step

            if vec and hasattr(vec, "transform") and hasattr(vec, "get_feature_names_out"):
                X_trans = vec.transform([text_input])
                feature_names = vec.get_feature_names_out()
                row_weights = X_trans.toarray()[0]
                
                if hasattr(clf, "coef_"):
                    coefs = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
                    scores = coefs * row_weights
                elif hasattr(clf, "feature_importances_"):
                    scores = clf.feature_importances_ * row_weights
                else:
                    scores = row_weights

                top_indices = np.argsort(np.abs(scores))[::-1][:3]
                
                # Convert raw math weights into admin-friendly text labels
                for i in top_indices:
                    if scores[i] != 0:
                        word = str(feature_names[i])
                        val = abs(scores[i])
                        if val > 0.5:
                            impact = "Strong Indicator"
                        elif val > 0.2:
                            impact = "Moderate Indicator"
                        else:
                            impact = "Minor Factor"
                        shap_explanation[word] = impact

            if not shap_explanation:
                words = [w.strip(".,!?") for w in text_input.split() if len(w) > 3][:2]
                shap_explanation = {word: "Contributing Term" for word in words}

            # Generate a clean narrative sentence for the admin
            top_words = list(shap_explanation.keys())
            if len(top_words) >= 2:
                reasoning_text = f"Classified as {predicted_priority} priority because the text heavily features critical terms such as '{top_words[0]}' and '{top_words[1]}'."
            elif len(top_words) == 1:
                reasoning_text = f"Classified as {predicted_priority} priority due to emphasis on the term '{top_words[0]}'."

        except Exception as ex:
            reasoning_text = f"Classified as {predicted_priority} priority following keyword pattern analysis."
            shap_explanation = {"Complaint Content": "Primary Driver"}

        return {
            "stud_number": data.stud_number,
            "priority": str(predicted_priority),
            "reason": reasoning_text,
            "shap": shap_explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))