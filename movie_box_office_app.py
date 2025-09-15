import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv(r"C:\Users\Dell\Desktop\ai\movie_box_office_dataset.csv")

le_genre = LabelEncoder()
le_month = LabelEncoder()

data["Genre"] = le_genre.fit_transform(data["Genre"])
data["Release_Month"] = le_month.fit_transform(data["Release_Month"])

X = data[['Genre', 'Budget', 'Cast_Popularity', 'Director_Popularity', 'Marketing_Spend', 'Release_Month']]
y = data['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("✅ Model Training Complete")
print("R² Score:", r2)
print("MSE:", mse)
print("MAE:", mae)

sample_pred = model.predict([[le_genre.transform(['Action'])[0], 80, 7, 6, 25, le_month.transform(['June'])[0]]])
print("Predicted Revenue for sample movie:", sample_pred)

with open("linear_movie_model.pkl", "wb") as f:
    pkl.dump((model, le_genre, le_month), f)

print("✅ Model saved as linear_movie_model.pkl")
