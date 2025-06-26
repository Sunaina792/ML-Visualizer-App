# regression_models.py

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred

def multiple_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred

def polynomial_regression(X, y, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred

def ridge_regression(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred

def lasso_regression(X, y, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred

def logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred


