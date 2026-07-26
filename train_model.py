import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Load dataset
df = pd.read_csv("Dataset/USA_Housing.csv")


# Rename columns
df = df.rename(columns={
    "Avg. Area Income": "avg_area_income",
    "Avg. Area House Age": "avg_area_house_age",
    "Avg. Area Number of Rooms": "avg_area_num_rooms",
    "Avg. Area Number of Bedrooms": "avg_area_num_bedrooms",
    "Area Population": "area_population",
    "Price": "price"
})


# Remove Address because it is not used by the model
df = df.drop(columns=["Address"])


# Feature engineering
df["rooms_per_bedroom"] = (
    df["avg_area_num_rooms"] /
    df["avg_area_num_bedrooms"]
)


# Select input features
features = [
    "avg_area_income",
    "avg_area_house_age",
    "avg_area_num_rooms",
    "avg_area_num_bedrooms",
    "area_population",
    "rooms_per_bedroom"
]

X = df[features]

# Target
y = df["price"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scale input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# Train Linear Regression model
model = LinearRegression()

model.fit(X_train_scaled, y_train)


# Save trained model and scaler
joblib.dump(model, "Model/linear_regression_model.pkl")
joblib.dump(scaler, "Model/scaler.pkl")


print("Model and scaler saved successfully.")