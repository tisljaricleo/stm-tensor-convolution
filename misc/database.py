"""Database tools

Script contains all required database tools regarding CRUD operations

***Functions***
-
-
-
-
-
"""

from misc import config
config.initialize_metadata()

__licence__ = config.LICENCE
__author__ = config.AUTHOR
__email__ = config.EMAIL
__status__ = config.STATUS
__docformat__ = config.DOCFORMAT


from pymongo import MongoClient


def init(db_name):
    """Creates client and database objects

    :param db_name: Database name
    :type db_name: str
    :return: Returns database and client objects
    """
    client = MongoClient('localhost', 27017)
    db = client[db_name]
    return db, client


def closeConnection(client):
    """Closes connection

    :param client: Opened client object
    :return:
    """
    client.close()


def insertMany(db, collection, data):
    """Insert more than one object

    :param db: Database object
    :param collection: Collection name
    :param data: Object to save in database
    :type collection: str
    :type data: list
    :return:
    """
    db[collection].insert_many(data)


def insertOne(db, collection, data):
    db[collection].insert_one(data)


def selectAll(db, collection):
    return list(db[collection].find())


def selectSome(db, collection, query):
    return list(db[collection].find(query))


def selectSkipLimit(db, collection, skip, limit):
    return list(db[collection].find().skip(skip).limit(limit))


def count(db, collection):
    return int(db[collection].find().count())


def groupBy(db, collection, query):
    return db[collection].aggregate([{'$group': query}])
