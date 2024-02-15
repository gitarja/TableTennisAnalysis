import numpy as np
import networkx as nx
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork as DBN
from Conf import single_summary_path
from FeaturesReader import SingleFeaturesReader
from pgmpy.inference import DBNInference
import matplotlib.pyplot as plt
model = DBN(
    [

        (("pr_p1_al_miDo", 0), ("ec_fs_ball_racket_dir", 0)),



        (("ec_fs_ball_racket_dir", 0), ("im_rb_ang_collision", 0)),
        (("ec_fs_ball_racket_dir", 0), ("im_rack_wrist_dist", 0)),
        (("ec_fs_ball_racket_dir", 0), ("im_rb_dist", 0)),



        (("im_rb_ang_collision", 0), ("pr_p1_al_miDo", 1)),




    ]
)

nx_graph = nx.DiGraph(model.edges())
nx.draw(nx_graph, with_labels=True)
plt.show()

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

df = pd.read_csv(single_summary_path)
df = df[(df["norm_score"] > 0.5) & (df["Tobii_percentage"] > 65)]
features_reader = SingleFeaturesReader(path, df["Subject1"].values, n_window=2, discretization=True)

X = features_reader.getAllData()

model_fit = model.fit(X)

dbn_inf = DBNInference(model)
for i in  range(10):
    print(dbn_inf.query([('ec_fs_ball_racket_dir', 0)], {('im_rb_ang_collision', 0):int(i)})[('ec_fs_ball_racket_dir', 0)].values)

