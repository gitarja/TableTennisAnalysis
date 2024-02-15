import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotSHAP(shap_values, x, columns, results_path, prefix=""):
    sns.set_theme()
    sns.set_style("white")
    # combing shap and x
    shap_columns = ["shap_"+c for c in columns]
    summary_df = pd.DataFrame(np.concatenate([x.values, shap_values], axis=-1), columns=columns + shap_columns)

    for c in columns:
        g = sns.scatterplot(x=c, y="shap_" + c, data=summary_df, s=7, color="#08519c", edgecolor = None, alpha=0.5)
        #
        # sns.kdeplot(x=c, y="shap_" + c, levels=5, color="black", linewidths=1, data=summary_df)

        # if len(np.unique(summary_df[c].values)) < 10:
        #     g = sns.scatterplot(x=c, y="shap_"+c, data=summary_df, s= 7, color= "#08519c")
        # else:
        #     g = sns.regplot(x=c, y="shap_"+c, data=summary_df, line_kws={"color": "#252525"} , scatter_kws={"s": 7, "color": "#08519c"}, order=4, x_jitter=.03)

        g.set(ylabel='SHAP values')
        plt.axhline(y=0., color="black", linestyle=":")

        sns.despine()
        # plt.show()
        plt.savefig(results_path + c +"_"+ prefix + ".png", format='png')
        plt.close()



def plotShapInteraction(shap_values, x, columns, results_path, prefix, ref, show_column):
    sns.set_theme()
    sns.set_style("white")
    # combing shap and x
    shap_columns = ["shap_"+c for c in columns]
    summary_df = pd.DataFrame(np.concatenate([x.values, shap_values], axis=-1), columns=columns + shap_columns)
    # summary_df["ec_start_fs"] = summary_df["ec_start_fs"] / np.max(summary_df["ec_start_fs"].values)
    summary_df = summary_df.loc[summary_df["ec_start_fs"] <=750]
    for c in show_column:
        g = sns.scatterplot(x=c, y="shap_" + c, data=summary_df, s=7, color="#08519c", edgecolor = None,  hue=ref, palette='coolwarm')

        g.set(ylabel='SHAP values')
        plt.axhline(y=0., color="black", linestyle=":")
        sns.despine()
        # plt.savefig(results_path + c +"_"+ prefix + ".png", format='png')
        plt.show()