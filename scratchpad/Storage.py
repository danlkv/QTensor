from mongocat import MongoCat
import json
from bson import ObjectId
from pathlib import Path


class FileStorage:
    """
    Writes data in .jsonl file.
    Makes sure object has _id prop
    """
    def __init__(self, collection, path='./data'):
        self.path = path
        self.filename = Path(self.path) / collection+'.jsonl'

    def put(self, object):
        """
        Generates ObjectId if no `_id` key is provided.
        Saves the object to file
        """
        if object.get('_id') is None:
            id_gen = str(ObjectId())
            object['_id'] = id_gen
            id_ = id_gen
        else:
            id_ = object['_id']

        line = json.dumps(object)
        with open(self.filename, 'a+') as f:
            f.write(line+'\n')

        return id_


class Storage:
    """
    A mongodb storage.
    - falls back to file storage if no connection
    - Generates id for every oject if no `_id` is provided

    If Mongo didn't fail, use self.mongo to read data
    """
    def __init__(self, collection, mongo_url,
                 database='qsim',
                 file_path='./data'
                ):
        try:
            mongocat = MongoCat(database, collection, mongo_url, parser='json')
            self.provider = mongocat
            self.mongo = mongocat.client

        except Exception as e:
            print('ERROR: failed to connect to Mongo:',e)
            self.provider = FileStorage(collection, path=file_path)
            print((f'Fallback to file storage in {self.storage.filename}.'
                   'Use mongocat to upload files when connected'
                  ))

    def put(self, object):
        """
        Saves the object, returns the id

        Parameters
        ----------
        object: dict
            object to store

        Returns
        -------
        str
            id of the object in the database

        """
        id = self.provider.put(object)
        return str(id)
