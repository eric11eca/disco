from pymongo import MongoClient

def get_database(dataset="snli", type="c2e"):
    """Returns the database instance for the given dataset and type
   
    :param dataset: the dataset to be used
    :param type: the type of the dataset
    :return: the database instance
    """
    CONNECTION_STRING = "localhost:27017"
    client = MongoClient(CONNECTION_STRING)
    collection = client['counterfactual'][f"{dataset}_{type}"]
    return collection

def query(collection, query):
    """Queries the database for the given query
    
    :param collection: the database instance
    :param query: the query to be used
    :return: the record
    """
    record = collection.find_one(query)
    if record is not None:
        del record["_id"]
    return record

def update(collection, query, data):
    """Updates the database for the given query and data
    
    :param collection: the database instance
    :param query: the query to be used
    :param data: the data to be used
    """
    collection.update_one(query, data)

def insert(collection, data):
    """Inserts the data into the database
    
    :param collection: the database instance
    :param data: the data to be used
    """
    collection.insert_one(data)