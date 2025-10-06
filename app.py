import streamlit as st
import pandas as pd
import numpy as np
import os
import json

st.set_page_config(page_title="Demo")

st.title("Outpeer Demo1")
st.write("Car Price Predictor")


class MylinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.is_fitted = False
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("X must be 1D or 2D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        X = X.astype(float)
        y = y.astype(float)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        ones = np.ones((n_samples, 1), dtype=float)
        X_b = np.hstack([ones, X])
        XtX = X_b.T @ X_b
        XtY = X_b.T @ y
        beta = np.linalg.pinv(XtX) @ XtY
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].reshape(-1)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("X must be 1D or 2D array.")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match the fitted model ({self.coef_.shape[0]})."
            )
        X = X.astype(float)
        return X @ self.coef_ + self.intercept_


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_to_clean = [
        "Engine Size (L)",
        "Horsepower",
        "Price (in USD)",
    ]
    for col in cols_to_clean:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace('"', '', regex=False)
            .str.replace("-", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[
        "Engine Size (L)",
        "Horsepower",
        "Price (in USD)",
    ])
    return df


def _encode_categories(df: pd.DataFrame):
    car_make_cats = df["Car Make"].astype("category").cat.categories.tolist()
    make_to_code = {v: i for i, v in enumerate(car_make_cats)}
    return make_to_code


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    make_to_code = _encode_categories(df)
    X_df = df.drop(columns=[c for c in ["Car Model", "Torque (lb-ft)", "0-60 MPH Time (seconds)"] if c in df.columns])
    X_df["Car Make"] = X_df["Car Make"].map(make_to_code)
    y_series = pd.to_numeric(X_df["Price (in USD)"], errors="coerce")
    X_df = X_df.drop(columns=["Price (in USD)"])

    # Ensure numeric types as in notebook after cleaning
    X_num = X_df.apply(pd.to_numeric, errors="coerce")
    valid_mask = X_num.notna().all(axis=1) & y_series.notna()
    X = X_num.loc[valid_mask].to_numpy(dtype=float)
    y = y_series.loc[valid_mask].to_numpy(dtype=float)

    feature_cols = list(X_num.columns)
    model = MylinearRegression()
    model.fit(X, y)
    meta = {
        "feature_cols": feature_cols,
        "make_to_code": make_to_code,
    }
    return model, meta


def save_artifacts(model: MylinearRegression, meta: dict, mse: float | None = None):
    os.makedirs("artifacts", exist_ok=True)
    # save model as json (coef and intercept)
    model_path = os.path.join("artifacts", "model.json")
    meta_path = os.path.join("artifacts", "meta.json")
    with open(model_path, "w") as f:
        json.dump({"intercept": model.intercept_, "coef": model.coef_.tolist()}, f)
    with open(meta_path, "w") as f:
        json.dump({**meta, "mse": mse}, f)
    if mse is not None:
        metrics_csv = os.path.join("artifacts", "metrics.csv")
        header_needed = not os.path.exists(metrics_csv)
        with open(metrics_csv, "a") as f:
            if header_needed:
                f.write("mse\n")
            f.write(f"{mse}\n")


def load_artifacts():
    model_path = os.path.join("artifacts", "model.json")
    meta_path = os.path.join("artifacts", "meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    with open(model_path, "r") as f:
        m = json.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    mdl = MylinearRegression()
    mdl.intercept_ = float(m["intercept"]) if m.get("intercept") is not None else 0.0
    mdl.coef_ = np.array(m.get("coef", []), dtype=float)
    mdl.is_fitted = True
    return mdl, meta


df = load_data("./datasets/sport_cars.csv")
model, meta = train_model(df)

car_make = st.selectbox("Car Make", sorted(df["Car Make"].unique()))

year = st.number_input("Year", min_value=1900, max_value=2100, value=int(df["Year"].median()))
engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=float(df["Engine Size (L)"].median()), step=0.1)
horsepower = st.number_input("Horsepower", min_value=0.0, value=float(df["Horsepower"].median()), step=10.0)

if st.button("Predict"):
    make_code = meta["make_to_code"].get(car_make, -1)
    row = {
        "Year": float(year),
        "Engine Size (L)": float(engine_size),
        "Horsepower": float(horsepower),
        "Car Make": float(make_code),
    }
    features = np.array([[row[c] for c in meta["feature_cols"]]], dtype=float)
    pred = float(model.predict(features)[0])
    st.success(f"The predicted price is ${pred:,.0f}")

with st.sidebar:
    st.subheader("MLOps")
    if st.button("Load latest model"):
        loaded, loaded_meta = load_artifacts()
        if loaded is not None:
            model = loaded
            meta = loaded_meta
            st.success("Loaded model from artifacts/")
        else:
            st.warning("No saved artifacts found.")

    if st.button("Save model"):
        # compute a quick MSE on training data for logging
        make_codes = df["Car Make"].map(meta["make_to_code"]).astype(float)
        X_log = pd.DataFrame({
            "Year": df["Year"].astype(float),
            "Engine Size (L)": df["Engine Size (L)"].astype(float),
            "Horsepower": df["Horsepower"].astype(float),
            "Car Make": make_codes,
        })
        X_log = X_log[meta["feature_cols"]].to_numpy(dtype=float)
        y_log = pd.to_numeric(df["Price (in USD)"], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(X_log).any(axis=1) & ~np.isnan(y_log)
        mse = float(((y_log[mask] - model.predict(X_log[mask])) ** 2).mean())
        save_artifacts(model, meta, mse)
        st.success(f"Saved model with MSE={mse:.2f}")

