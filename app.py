from flask import Flask, request, render_template
import pickle
import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load models and API keys
model = pickle.load(open('Model/Housing_Model', 'rb'))
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def generate_caption(image_path, message_text):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    message = HumanMessage(
        content=[
            {"type": "text", "text": message_text},
            {"type": "image_url", "image_url": image_path}
        ]
    )
    result = llm.invoke([message])
    return result.content

@app.route('/', methods=['GET', 'POST'])
def home():
    price_result = None
    caption = None

    if request.method == 'POST':
        # Handle image and form details
        try:
            # House details
            query_tradetime = float(request.form['tradetime'])
            query_followers = int(request.form['followers'])
            query_square = float(request.form['square'])
            query_livingroom = int(request.form['livingroom'])
            query_drawingroom = int(request.form['drawingroom'])
            query_kitchen = int(request.form['kitchen'])
            query_bathroom = int(request.form['bathroom'])
            query_constructiontime = float(request.form['constructiontime'])
            query_communityaverage = float(request.form['communityaverage'])
            query_renovationcondition = request.form['renovationcondition']
            query_buildingstructure = request.form['buildingstructure']
            query_elevator = request.form['elevator']
            image = request.files['image']
            message_text = request.form['crisp_description']

            # Handle categorical variables
            renovationCondition_2, renovationCondition_3, renovationCondition_4 = 0, 0, 0
            if query_renovationcondition == "renovationCondition_2":
                renovationCondition_2 = 1
            elif query_renovationcondition == "renovationCondition_3":
                renovationCondition_3 = 1
            elif query_renovationcondition == "renovationCondition_4":
                renovationCondition_4 = 1

            buildingStructure_2, buildingStructure_3, buildingStructure_4, buildingStructure_5, buildingStructure_6 = 0, 0, 0, 0, 0
            if query_buildingstructure == "buildingStructure_2":
                buildingStructure_2 = 1
            elif query_buildingstructure == "buildingStructure_3":
                buildingStructure_3 = 1
            elif query_buildingstructure == "buildingStructure_4":
                buildingStructure_4 = 1
            elif query_buildingstructure == "buildingStructure_5":
                buildingStructure_5 = 1
            elif query_buildingstructure == "buildingStructure_6":
                buildingStructure_6 = 1

            elevator_1 = 1 if query_elevator == "elevator_1" else 0

            # Predict price
            model_data = [[
                query_tradetime, query_followers, query_square, query_livingroom,
                query_drawingroom, query_kitchen, query_bathroom, query_constructiontime,
                query_communityaverage, renovationCondition_2, renovationCondition_3, renovationCondition_4,
                buildingStructure_2, buildingStructure_3, buildingStructure_4, buildingStructure_5,
                buildingStructure_6, elevator_1
            ]]
            result = model.predict(model_data)
            price_result = "{:.3f}".format(float(result))

            # Generate caption
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            caption = generate_caption(image_path, message_text)
        
        except ValueError:
            price_result = "Invalid input provided for housing prediction."

    return render_template('index.html', price_result=price_result, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
