from insight_data import ai_quotes
from flask_restful import Resource, reqparse
import random

class Quote(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        params = parser.parse_args()
        id = params["id"]

        if id == 0:
            return random.choice(ai_quotes), 200

        for quote in ai_quotes:
            if quote["id"] == id:
                return quote, 200

        return "Quote not found", 404

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("author")
        parser.add_argument("quote")
        parser.add_argument("id")
        params = parser.parse_args()
        id = params["id"]

        for quote in ai_quotes:
            if id == quote["id"]:
                return f"Quote with id {id} already exists", 400

        quote = {
            "id": int(id),
            "author": params["author"],
            "quote": params["quote"]
        }

        ai_quotes.append(quote)

        return quote, 201

    def put(self):
        parser = reqparse.RequestParser()
        parser.add_argument("author")
        parser.add_argument("quote")
        params = parser.parse_args()
        id = params["id"]

        for quote in ai_quotes:
            if (id == quote["id"]):
                quote["author"] = params["author"]
                quote["quote"] = params["quote"]
                return quote, 200

        quote = {
            "id": id,
            "author": params["author"],
            "quote": params["quote"]
        }

        ai_quotes.append(quote)
        return quote, 201

    def delete(self):
        global ai_quotes
        parser = reqparse.RequestParser()
        parser.add_argument("id")
        params = parser.parse_args()
        id = params["id"]

        ai_quotes = [qoute for qoute in ai_quotes if qoute["id"] != id]
        return f"Quote with id {id} is deleted.", 200
