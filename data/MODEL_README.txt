Final model pipeline: final_mlp_pipeline.pkl

How to load and predict:
import joblib
pipe = joblib.load('final_mlp_pipeline.pkl')
proba = pipe.predict_proba(x_new)[:,1]
pred = pipe.predict(x_new)

Saved metrics file: final_model_metrics.json