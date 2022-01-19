import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns


def main():
    datasets = ["mnist", "fmnist", "cifar10"]
    selected_epochs = [1,4,10]
    # k_neighbors = [10, 15, 20]
    k_neighbors = [15]
    col = np.array(["dataset", "method", "type", "hue", "period", "eval"])
    df = pd.DataFrame({}, columns=col)

    for i in range(3): # dataset
        dataset = datasets[i]
        data = np.array([])
        # load data from evaluation.json
        content_path = "/home/xianglin/projects/DVI_data/TemporalExp/resnet18_{}".format(dataset)
        for epoch_id in range(3):
            epoch = selected_epochs[epoch_id]
            eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_step2_A.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            inv_acc_train = round(eval["inv_acc_train"], 3)
            inv_acc_test = round(eval["inv_acc_test"], 3)

            if len(data)==0:
                data = np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(str(epoch_id)), inv_acc_train]])
            else:
                data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(str(epoch_id)), inv_acc_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-Test", "{}".format(str(epoch_id)), inv_acc_test]])), axis=0)
        
        eval_path = "/home/xianglin/projects/DVI_data/TemporalExp/resnet18_{}/Model/test_evaluation.json".format(dataset)
        with open(eval_path, "r") as f:
                eval = json.load(f)
        for epoch_id  in range(3):
            epoch = selected_epochs[epoch_id]
            ppr_train = round(eval["ppr_train"][str(epoch)], 3)
            ppr_test = round(eval["ppr_test"][str(epoch)], 3)

            data = np.concatenate((data, np.array([[dataset, "TimeVis", "Train", "TimeVis-Train",  "{}".format(str(epoch_id)), ppr_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "TimeVis", "Test", "TimeVis-Test", "{}".format(str(epoch_id)), ppr_test]])), axis=0)

        # df_tmp = pd.DataFrame(data, columns=col)
        # df = df.append(df_tmp, ignore_index=True)
        # df[["period"]] = df[["period"]].astype(int)
        # df[["eval"]] = df[["eval"]].astype(float)

        df_tmp = pd.DataFrame(data, columns=col)
        df = df.append(df_tmp, ignore_index=True)
        df[["period"]] = df[["period"]].astype(int)
        # df[["k"]] = df[["k"]].astype(int)
        df[["eval"]] = df[["eval"]].astype(float)

    #%%
    df.to_excel("PPR.xlsx")
    pal20c = sns.color_palette('tab20c', 20)
    sns.set_theme(style="whitegrid", palette=pal20c)
    hue_dict = {
            "DVI-Train": pal20c[0],
            "TimeVis-Train": pal20c[4],

            "DVI-Test": pal20c[3],
            "TimeVis-Test": pal20c[7],
        }
    sns.palplot([hue_dict[i] for i in hue_dict.keys()])

    axes = {'labelsize': 9,
            'titlesize': 9,}
    mpl.rc('axes', **axes)
    mpl.rcParams['xtick.labelsize'] = 9

    hue_list = ["DVI-Train", "DVI-Test", "TimeVis-Train", "TimeVis-Test"]

    fg = sns.catplot(
        x="period",
        y="eval",
        hue="hue",
        hue_order=hue_list,
        # order = [1, 2, 3, 4, 5],
        # row="method",
        col="dataset",
        ci=0.001,
        height=2.5, #2.65,
        aspect=1.0,#3,
        data=df,
        kind="bar",
        palette=[hue_dict[i] for i in hue_list],
        legend=True
    )
    sns.move_legend(fg, "lower center", bbox_to_anchor=(.42, 0.92), ncol=3, title=None, frameon=False)
    mpl.pyplot.setp(fg._legend.get_texts(), fontsize='9')

    axs = fg.axes[0]
    max_ = df["eval"].max()
    # min_ = df["eval"].min()
    axs[0].set_ylim(0., max_*1.1)
    axs[0].set_title("MNIST")
    axs[1].set_title("FMNIST")
    axs[2].set_title("CIFAR-10")

    (fg.despine(bottom=False, right=False, left=False, top=False)
     .set_xticklabels(['Begin', 'Mid','End'])
     .set_axis_labels("", "PPR")
     )
    # fg.fig.suptitle("Prediction Preserving property")

    fg.savefig(
        "inv_accu.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )



if __name__ == "__main__":
    main()

