
import pickle

class DataLoader():
    def __init__(self, dataset:str):
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        self.train_file_path = f"/home/jinsoo/2025_SCI/DFGAT_addRating/MyGAT/Data/{dataset}_trn.pkl"
        self.val_file_path = f"/home/jinsoo/2025_SCI/DFGAT_addRating/MyGAT/Data/{dataset}_val.pkl"
        self.test_file_path = f"/home/jinsoo/2025_SCI/DFGAT_addRating/MyGAT/Data/{dataset}_tst.pkl"

    @staticmethod
    def _cast_timestamp(data):
        
        # 레이팅 정보가 있는 데이터셋들
        if len(data[0]) == 4:
            return [[src, dst, int(ts), float(rating)] for src, dst, ts, rating in data]
        # 레이팅 정보가 없는 데이터셋 (Gowalla) [uid, lid, timestamp]
        else:
            return [[src, dst, int(ts)] for src, dst, ts in data]

    def load_dataset(self):
        
        with open(self.train_file_path, 'rb') as fp:
            self.train_set = pickle.load(fp)
        
        with open(self.val_file_path, 'rb') as fp:
            self.val_set = pickle.load(fp)

        with open(self.test_file_path, 'rb') as fp:
            self.test_set = pickle.load(fp)

        self.train_set = self._cast_timestamp(self.train_set)
        self.val_set   = self._cast_timestamp(self.val_set)
        self.test_set  = self._cast_timestamp(self.test_set)

        return (self.train_set, self.val_set, self.test_set)
   