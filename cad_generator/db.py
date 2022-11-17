from pymongo import MongoClient

def get_database(dataset="snli"):
   CONNECTION_STRING = "localhost:27017"
   client = MongoClient(CONNECTION_STRING)
   collection = client['user_shopping_list'][dataset]
   return collection

def query(collection, query):
    record = collection.find_one(query)
    if record is not None:
        del record["_id"]
    return record

def update(collection, query, data):
    collection.update_one(query, data)

def insert(collection, data):
    collection.insert_one(data)