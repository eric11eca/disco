from pymongo import MongoClient

def create_collection(cache, collection_name):
    """Creates a collection in the database"""
    try:
        cache.createCollection(collection_name)
    except Exception as e:
        pass

def get_database(dataset="snli", template_name="masked_cad_premise"):
    """Returns the database instance for the given dataset and template_name
   
    :param dataset: the dataset to be used
    :param template_name: the type of the template
    :return: the database instance
    """
    CONNECTION_STRING = "localhost:27017"
    client = MongoClient(CONNECTION_STRING)
    collection = client.get_database('disco')

    gen_name = f"{dataset}_{template_name}_gen"

    create_collection(collection, gen_name)
    output_cache = collection[gen_name]
    return output_cache

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


def delete(collection, query):
    """Deletes the record from the database
    
    :param collection: the database instance
    :param query: the query to be used
    """
    collection.delete_one(query)


def get_all(collection):
    """Returns all the records from the database
    
    :param collection: the database instance
    :return: the records
    """
    records_ls = []
    records = collection.find({ })
    for record in records:
        del record['_id']
        records_ls.append(record)
    return records_ls

def get_all_accepted(collection):
    """
    Returns all the accepted records from the database

    :param collection: the database instance
    :return: the accepted records
    """
    records_ls = []
    records = collection.find({ "accept": True })
    for record in records:
        del record['_id']
        records_ls.append(record)
    return records_ls

def group_by_key(collection, key="guid"):
    agg_result = collection.aggregate([{
        "$group": {"_id" : f"${key}", 
        "count": {"$count" : {}}
    }}])
    grouped = [result for result in agg_result]
    return grouped