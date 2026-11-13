import pandas as pd
# Import dataset into a Pandas DataFrame
dataset = pd.read_csv("SHA09.csv")
dataset.columns = [c.strip() for c in dataset.columns]
# Filter for Household Out-of-Pocket Payments and All Revenues
dataset_filtered = dataset[
    (dataset["Type of Financing Scheme"].str.contains("Household Out-of-Pocket Payments", na=False)) &
    (dataset["Revenue of Financing Scheme"] == "All Revenues (FS.1 - FS.6)")
].copy()
# Clean missing values and calculate Monthly Medical Reserve per person (€)
dataset_clean = dataset_filtered[["Year", "VALUE"]].dropna().sort_values("Year")
irish_population = 5100000
dataset_clean["Monthly_Medical_Reserve"] = (dataset_clean["VALUE"] * 1000000 / irish_population) / 12
print("Raw Dataset Shape:", dataset.shape)
print("Cleaned Target Dataset Shape:", dataset_clean.shape)
dataset.info()
dataset_clean.info()
dataset_clean.describe()
# Find correlations with your target column
print(dataset_clean.corr()["Monthly_Medical_Reserve"].abs().sort_values(ascending=False))
import matplotlib.pyplot as plt
# Exploratory Data Analysis (EDA) Visualisation
plt.figure(figsize=(9, 5))
plt.plot(
    dataset_clean["Year"],
    dataset_clean["Monthly_Medical_Reserve"],
    marker="o",
    color="#1f77b4",
    linewidth=2.5,
    markersize=7,
    label="Monthly Out-of-Pocket Reserve (€)"
)
plt.title("EDA: Monthly Out-of-Pocket Healthcare Cost per Person in Ireland (2011–2024)", fontsize=12, fontweight="bold")
plt.xlabel("Year", fontsize=11)
plt.ylabel("Monthly Cost per Person (€)", fontsize=11)
plt.xticks(dataset_clean["Year"], rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define feature (X = Year) and target (y = Monthly_Medical_Reserve)
X = dataset_clean[["Year"]]
y = dataset_clean["Monthly_Medical_Reserve"]

# Split into Training and Testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
# Linear Regression algorithm
model = LinearRegression()

#  Train the model on the Training dataset
model.fit(X_train, y_train)

