from flask import Flask, render_template, request
import joblib
'''flask - a lightweight python frame work to build web apps
render_template - loads html from the templates/ folder
req -  gets data from the user(via form)
joblib - used to load your trained Ml model and vectorizer 
'''
app = Flask(__name__)

model = joblib.load("severity_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    '''the function runs when someone opens the root URL
    GET-allows displaying the form
    POST-receiving the data '''
    prediction = ""
    if request.method == "POST":
        bug_text = request.form["bug_description"]
        if bug_text:
            bug_lines = bug_text.strip().split("\n")
            bug_vectors = vectorizer.transform(bug_lines)
            predictions = model.predict(bug_vectors)
            prediction = "<br>".join(
                 f" <b>{line}</b> → <span style='color:blue'>{pred}</span>"
                 for line, pred in zip(bug_lines, predictions)
            )
    return render_template("index.html", prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True, port=8080)
