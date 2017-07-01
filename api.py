"""import of flask"""
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from bson.json_util import dumps

try:
    import simplejson as json
except ImportError:
    try:
        import json
    except ImportError:
        raise ImportError

import datetime
from bson.objectid import ObjectId
from werkzeug import Response

class MongoJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return unicode(obj)
        return json.JSONEncoder.default(self, obj)

def mongoToJson(*args, **kwargs):
    """ jsonify with support for MongoDB ObjectId
    """
    return Response(json.dumps(*args, cls=MongoJsonEncoder), mimetype='application/json')

app = Flask(__name__)
app.config['MONGO_DBNAME'] = 'test'

mongo = PyMongo(app)

@app.route("/")
def index():
    """Index of the api"""
    return jsonify({
        'message': 'Hello, this is the Root route of the API, call /songs to get songs'
    })

@app.route("/songs")
def songs():
    """List of songs route"""
    data = mongo.db.songs
    output = [song for song in data.find()]
    return mongoToJson(output)

@app.route("/song", methods=["POST", "PUT"])
def song():
    """Register a song"""
    data = mongo.db.test
    if request.method == "POST":
        print request.json["song_name"]
        data.insert(request.json)
    return jsonify(message="the song is registered")

if __name__ == "__main__":
    app.run()
