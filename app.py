from flask import Flask, render_template, request
import model as m

app = Flask(__name__)

@app.route("/")
def model():
    if request.method == "POST":
        formula = request.form["formula"]
        #formula = [formula]
        bulk = m.bulk_modulus_prediction(formula)
        xyz = bulk

    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        formula = request.form["formula"]
        #formula = [formula]
        bulk = m.bulk_modulus_prediction(formula)
        xyz = bulk

    return render_template("index.html", name = xyz)


if __name__ == "__main__":
    app.run(debug=True)
