from collections import defaultdict
import csv
import os
import pickle

class MovieLens25MPreprocessor():
    def __init__(self):
        self.movieIdxRemapper = defaultdict(int)
        self.userIdxRemapper = defaultdict(int)

        # # 리매핑됭 uid를 이용하여 타임스탬프를 조회하기 위한 딕셔너리
        # self.remapUid2timestamps = defaultdict(list)

        self.movies_path = "./ml-25m/movies.csv"
        self.ratings_path = "./ml-25m/ratings.csv"

        self.movie_datas = []
        self.edge_datas = []

    # 1
    def load_movie_data(self):
        """
            movies_path.csv 데이터셋

            movieId,title,genres
            1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
            ...
        """

        with open(self.movies_path, 'r', encoding='utf8') as fp:
            reader = csv.reader(fp)
            next(reader)  # 헤더 제거
            for data in reader:
                mid, title, genres = data[0], data[1], data[2]
                self.movie_datas.append([mid, title, genres])
    # 2
    def load_edge_data(self):
        """
            ratings_path.csv 데이터셋

            userId,movieId,rating,timestamp
            1,1,4.0,964982703
            1,3,4.0,964981247
        """

        with open(self.ratings_path, 'r', encoding="utf8") as fp:
            reader = csv.reader(fp)
            next(reader)  # 헤더 제거

            for data in reader:
                uid, mid, rating, timestamp = data[0], data[1], data[2], data[3]
                self.edge_datas.append([uid, mid, rating, int(timestamp)])

    # 3 ID를 0번부터 순차적으로 매핑하는 함수
    def make_remapper_dict(self):
        for idx, movie_data in enumerate(self.movie_datas):
            original_mid = movie_data[0] # mid
            self.movieIdxRemapper[original_mid] = idx
            movie_data[0] = idx

        user_set = set()
        for edge_data in self.edge_datas:
            user_set.add(edge_data[0])

        for idx, original_uid in enumerate(sorted(user_set)):
            self.userIdxRemapper[original_uid] = idx

        for edge_data in self.edge_datas:
            edge_data[0] = self.userIdxRemapper[edge_data[0]]  # user remap (0번부터 ~ )
            edge_data[1] = self.movieIdxRemapper[edge_data[1]] # movie remap (0번부터 ~)
    
    # 4 전체 엣지셋을 pkl로 저장하는 함수
    def save_edge_pkl(self):
        edges = []

        for edge_data in self.edge_datas:
            src = edge_data[0]
            dst = edge_data[1]

            edges.append([int(src),int(dst)])

        with open("edges_small.pkl", 'wb') as fp:
            pickle.dump(edges, fp)
    
    # 5 Train / Test로 나누어 PKL을 저장하는 함수
    def split_save_pkl(self):

        # train 데이터 비율
        TRAIN_RATIO = 0.6
        VAL_RATIO = 0.2
        TEST_RATIO = 0.2

        assert (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) == 1.0

        # key -> 유저 ID | val -> 해당 유저의 edge
        uid2edges = defaultdict(list)
        
        # key -> 유저 ID | val -> 해당 유저의 train / test edge
        train_edges = []
        val_edges = []
        test_edges = []

        for edge_data in self.edge_datas:
            uid, mid, rating, timestamp = edge_data[0], edge_data[1], edge_data[2], edge_data[3]
            uid2edges[uid].append([mid, timestamp, rating])
        
        # 유저별 영화 기록을, timestamp 기준으로 정렬
        for uid in uid2edges:
            uid2edges[uid].sort(key=lambda x:x[1])

        for uid in uid2edges:
            full_data = uid2edges[uid]

            if len(full_data) <= 5:
                print(f"5개 이하 데이터 무시함 uid -> {uid}")
                continue

            train_idx = int(len(full_data) * TRAIN_RATIO)
            val_idx   = int(len(full_data) * (TRAIN_RATIO + VAL_RATIO))

            # 훈련셋
            for mid, timestamp, rating in full_data[:train_idx]:
                train_edges.append([uid, mid, timestamp, rating])

            # 검증셋
            for mid, timestamp, rating in full_data[train_idx:val_idx]:
                val_edges.append([uid, mid, timestamp, rating])

            # 테스트셋
            for mid, timestamp, rating in full_data[val_idx:]:
                test_edges.append([uid, mid, timestamp, rating])

        # 파일로 저장 (이름은 필요한 대로 변경)
        with open("movielens_25M_trn.pkl", 'wb') as fp:
            pickle.dump(train_edges, fp)
        with open("movielens_25M_val.pkl", 'wb') as fp:
            pickle.dump(val_edges, fp)
        with open("movielens_25M_tst.pkl", 'wb') as fp:
            pickle.dump(test_edges, fp)


    def preprocess(self):
        self.load_movie_data()
        self.load_edge_data()
        self.make_remapper_dict()
        # self.save_pkl()
        self.split_save_pkl()

        print("완료")
        # print("pkl 파일 저장 형태")
        # for data in self.edge_datas[-10:]:
        #     print(data)


if __name__ == "__main__":
    pr = MovieLens25MPreprocessor()
    pr.preprocess()