import numpy as np
from joblib import dump,load
model = load('Real_Estate.joblib')


features = np.array([[-4.04805497e-01,  2.88894949e+00, -1.39324480e+00,
        -2.72888411e-01, -1.29089824e+00,  1.40140069e+00,
        -1.24839122e+00,  1.65332037e+00, -8.52752811e-01,
        -4.44929808e-01, -2.60558671e+00,  4.28667786e-01,
        -1.19447262e+00]])

print(model.predict(features))
 