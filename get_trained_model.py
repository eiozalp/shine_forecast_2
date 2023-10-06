import joblib
from model import Model

model = Model()
model.split()
model.fit()


joblib.dump(model, "model.joblib")