




1.
"C:\Program Files\MongoDB\Server\4.2\bin\mongod.exe" --dbpath D:\data\db\

2.
"C:\Program Files\MongoDB\Server\4.2\bin\mongo.exe"

3.
db.adminCommand( { listDatabases: 1 } )

4.
use SpeedTransitionDB

5.
db.getCollectionNames()
> db.getCollectionNames()
[
        "routes",
        "routesNEW",
        "spatialMatrix5080abs",
        "spatialMatrix50abs",
        "spatialMatrix50rel",
        "spatialMatrixNEWrel",
        "spatialMatrixRWLNEWrel",
        "transitions",
        "transitionsNEW"
]

6.
db.transitions.find().limit(1).pretty()
db.transitions.find().skip(5)
db.collection.count()

7.
db.collection.createIndex({a: 1, b: 1})
db.collection.getIndexes()

8.
db.spatialMatrixRWLNEWrel.find({origin_id: 201690}).explain("executionStats")