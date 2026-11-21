import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ૧. Kaggle ની અસલી CSV ફાઈલ લોડ કરવી
df = pd.read_csv("Advertising.csv")

# CSV માં કયા કોલમ છે તે ચેક કરવા (ઘણીવાર 'Unnamed: 0' નામનો વધારાનો કોલમ હોય છે)
print("કોલમના નામ:", df.columns.tolist())

# ૨. Input (X) અને Output (y) નક્કી કરવા (તમારી CSV ના કોલમ નામો મુજબ)
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# ૩. ડેટાને Training (80%) અને Testing (20%) માં વહેંચવો
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ૪. મોડલ ટ્રેઈન કરવું
model = LinearRegression()
model.fit(X_train, y_train)

# ૫. મોડલની ચોકસાઈ (Accuracy) ચેક કરવી
predictions = model.predict(X_test)
print("Model Accuracy (R2 Score):", round(r2_score(y_test, predictions) * 100, 2), "%")

# ૬. નવી અપડેટ થયેલી 'sales_model.pkl' ફાઈલ સેવ કરવી
joblib.dump(model, "sales_model.pkl")
print("અસલી ડેટા પર મોડલ ટ્રેઈન થઈને સેવ થઈ ગયું છે!")