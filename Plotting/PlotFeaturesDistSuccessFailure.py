import pandas as pd
from Conf import x_double_features_column, normalize_x_double_episode_columns, y_episode_column
import matplotlib.pyplot as plt
import seaborn as sns
from Double.FeatureReader import DoubleFeaturesReader
from Conf import double_summary_path

sns.set_theme()
sns.set_style("white")

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double_episode_features.pkl"

df = pd.read_csv(double_summary_path)
df = df[(df["Tobii_percentage"] > 65)]
features_reader = DoubleFeaturesReader(path, df["file_name"].values, n_window=4)

features = features_reader.getDF()

sns.displot(
    data=features, x="gc_p23_crossCorrVel", hue="success",
)

plt.show()