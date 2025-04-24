import pickle
import torch
import dgl

class GraphMaker():
    def __init__(self):
        self.dataset = None
        self.graph = None

        # 영화, 유저 또는 유저, 지역 등 src, dst 노드 타입
        self.node_type1 = []
        self.node_type2 = []

        # GOWALLA 데이터셋
        self.self_type = []

        self.num_type1_node = None
        self.num_type2_node = None

    # 만약 마지막 번호를 가진 사람이 평가를 안했으면, 해당 유저는 전체 그래프 노드에 포함되지 않는다. -> 이는 테스트 할 데이터도 없으므로 무시해도 OK.
    def load_file(self):

        with open(f"/home/jinsoo/2025_SCI/DFGAT_addRating/MyGAT/Data/{self.dataset}_trn.pkl", 'rb') as fp:
            train_data = pickle.load(fp)
        with open(f"/home/jinsoo/2025_SCI/DFGAT_addRating/MyGAT/Data/{self.dataset}_val.pkl", 'rb') as fp:
            val_data = pickle.load(fp)
        with open(f"/home/jinsoo/2025_SCI/DFGAT_addRating/MyGAT/Data/{self.dataset}_tst.pkl", 'rb') as fp:
            test_data = pickle.load(fp)

        full_data = train_data + val_data + test_data
        user_all = [row[0] for row in full_data]
        item_all = [row[1] for row in full_data]

        user_trn = [row[0] for row in train_data]
        item_trn = [row[1] for row in train_data]

        self.num_type1_node = max(user_all) + 1 
        self.num_type2_node = max(item_all) + 1

        graph_edges = {
            ('user', 'go', 'item'): (torch.tensor(user_trn), torch.tensor(item_trn)),
            ('item', 'back', 'user') : (torch.tensor(item_trn), torch.tensor(user_trn)),
            
            # self-loop 추가
            # ('user', 'selftype1','user') : (torch.arange(user_trn), torch.arange(item_trn)),
            # ('item', 'selftype2','item') : (torch.arange(item_trn), torch.arange(user_trn))
        }

        self.graph = dgl.heterograph(
            graph_edges,
            num_nodes_dict={
                'user': self.num_type1_node,
                'item': self.num_type2_node
            }
        )

    def get_graph(self, dataset:str):
        self.graph = None

        self.node_type1 = []
        self.node_type2 = []

        self.self_type = []
    
        self.dataset = dataset

        self.load_file()
        
        return self.graph

    def get_num_nodes(self):
        return (self.num_type1_node, self.num_type2_node)

        



