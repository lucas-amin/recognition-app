from flask import Flask
from flask_restful import Api
import random
from Quote import Quote

app = Flask(__name__)
api = Api(app)

api.add_resource(Quote, "/ai-quotes", "/ai-quotes/", "/ai-quotes/<int:id>")
if __name__ == '__main__':
    app.run(debug=True)