import pymongo as pm


class DbConnect:

    def __init__(self):
        self.client = pm.MongoClient('mongodb://localhost:27017/')
        self.db = self.client.Thesis
    
    
    

    
    
        
    

