
import pickle

class DataLoader():
    def __init__(self, dataset:str):
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        self.train_file_path = f"/home/jinsoo/2025_SCI/DFGAT/베이스라인/Data/{dataset}_trn.pkl"
        self.val_file_path = f"/home/jinsoo/2025_SCI/DFGAT/베이스라인/Data/{dataset}_val.pkl"
        self.test_file_path = f"/home/jinsoo/2025_SCI/DFGAT/베이스라인/Data/{dataset}_tst.pkl"

    def load_dataset(self):
        
        with open(self.train_file_path, 'rb') as fp:
            self.train_set = pickle.load(fp)
        
        with open(self.val_file_path, 'rb') as fp:
            self.val_set = pickle.load(fp)

        with open(self.test_file_path, 'rb') as fp:
            self.test_set = pickle.load(fp)

        return (self.train_set, self.val_set, self.test_set)
   