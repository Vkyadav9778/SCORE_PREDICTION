from flask import Flask,render_template,request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
app = Flask(__name__)
df=pd.read_csv("Updated_Score_file.csv")
# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.isnull().sum())


# SPLIT INPUT(X) AND OUTPUT(Y)
x=df[["Physics_Hours","Chemistry_Hours","Maths_Hours","English_Hours","Biology_Hours"]]
y=df["Total_Score"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# MODEL TRAINING
model = LinearRegression()

# FITTING THE MODEL
model.fit(x_train, y_train)#model fit

@app.route('/',methods=['GET','POST'])
def about():
    score = None
    if request.method == 'POST':
        physics = float(request.form['physics'])
        chemistry = float(request.form['chemistry'])
        maths = float(request.form['maths'])
        english = float(request.form['english'])
        biology = float(request.form['biology'])
        score = round(model.predict([[physics, chemistry, maths, english, biology]])[0], 2)
    return render_template('Dashboard.html', score=score)


if __name__ == '__main__':
    app.run(debug=True)


