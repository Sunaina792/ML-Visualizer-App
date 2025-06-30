# regression_models.py
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    return model, pred.flatten(), mse, r2



def multiple_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)


def polynomial_regression(X, y, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)


def ridge_regression(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)


def lasso_regression(X, y, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)


def svr_regression(X, y, kernel='rbf'):
    model = SVR(kernel=kernel)
    model.fit(X, y.values.ravel())
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)


def random_forest_regression(X, y, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X, y.values.ravel())
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)


def decision_tree_regression(X, y):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)


def knn_regression(X, y, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred, mean_squared_error(y, pred), r2_score(y, pred)