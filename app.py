# app.py

from flask import Flask,request, render_template, jsonify
from src.RestaurantRatingPrediction.pipelines.prediction_pipeline import CustomData, PredictPipeline

app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"], endpoint='predict_datapoint') 
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data = CustomData(
            online_order=str(request.form.get('online_order')),
            book_table=str(request.form.get('book_table')),
            votes=float(request.form.get('votes')),
            location=str(request.form.get('location')),
            rest_type=str(request.form.get('rest_type')),
            cuisines=str(request.form.get('cuisines')),
            cost_for_2=float(request.form.get('cost_for_2')),
            type=str(request.form.get('type'))
        )

        # this is my final data
        final_data = data.get_data_as_dataframe()        
        predict_pipeline = PredictPipeline()        
        pred = predict_pipeline.predict(final_data)        
        result = round(pred[0], 2)
        
        return render_template("result.html", final_result=result)


# execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
