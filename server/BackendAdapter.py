'''This class serves as a intermediate layer for tensorboard frontend and timeVis backend'''
import os
import json
import pandas as pd
import numpy as np


class TimeVisBackend:
    def __init__(self, data_provider, trainer, evaluator) -> None:
        self.data_provider = data_provider
        self.trainer = trainer
        self.evaluator = evaluator

    def subject_model_table(self):
        """get the dataframe for subject model table

        Returns
        -------
        df: pandas data frame
        """
        path_list = []
        epoch_list = []
        train_accu = []
        test_accu = []
        for n_epoch in range(self.data_provider.s, self.data_provider.e+1, self.data_provider.p):
            path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(n_epoch), "subject_model.pth")
            path_list.append(path)
            epoch_list.append(n_epoch)
            train_accu.append(self.data_provider.training_accu(n_epoch))
            test_accu.append(self.data_provider.testing_accu(n_epoch))
        df_dict = {
            "location": path_list,
            "epoch": epoch_list,
            "train_accu": train_accu,
            "test_accu": test_accu
        }
        df = pd.DataFrame(df_dict, index=pd.Index(range(len(path_list)), name="idx"))
        return df

    # Visualization model table
    def vis_model_table(self):
        """get the dataframe for vis model table
        """
        # TODO current implementation fit DVI but not timeVis, need to deprecate some useless fields...
        temporal = 1
        vis_path = self.data_provider.model_path

        path_list = []
        epoch_list = []
        temporal_list = []

        nn_train = []
        boundary_train = []
        ppr_train = []
        tnn_train = []
        st_nn_train = []

        nn_test = []
        boundary_test = []
        ppr_test = []
        tnn_test = []
        st_nn_test = []

        # placeholder
        ccr_train = []
        ccr_test = []

        eval_path = os.path.join(self.data_provider.model_path, "test_evaluation.json")
        with open(eval_path, "r") as f:
            eval = json.load(f)

        for n_epoch in range(self.data_provider.s, self.data_provider.e + 1, self.data_provider.p):
            path_list.append(vis_path)
            epoch_list.append(n_epoch)
            temporal_list.append(temporal)

            nn_train.append(eval["15"]["nn_train"][str(n_epoch)])
            nn_test.append(eval["15"]["nn_test"][str(n_epoch)])
            boundary_train.append(eval["15"]["b_train"][str(n_epoch)])
            boundary_test.append(eval["15"]["b_test"][str(n_epoch)])
            ppr_train.append(eval["ppr_train"][str(n_epoch)])
            ppr_test.append(eval["ppr_test"][str(n_epoch)])

        df_dict = {
            "location": path_list,
            "epoch": epoch_list,
            "temporal_loss": temporal_list,
            "nn_train": nn_train,
            "nn_test": nn_test,
            "boundary_train": boundary_train,
            "boundary_test": boundary_test,
            "ppr_train": ppr_train,
            "ppr_test": ppr_test,
            "ccr_train":ccr_train,
            "ccr_test": ccr_test
        }
        df = pd.DataFrame(df_dict, index=pd.Index(range(len(path_list)), name="idx"))
        return df

    # Sample table
    def sample_table(self):
        """
        sample table:
            label
            index
            type:["train", "test", others...]
            customized attributes
        :return:
        """
        train_labels = self.data_provider.train_labels().tolist()
        test_labels = self.data_provider.test_labels().tolist()
        labels = train_labels + test_labels
        train_type = ["train" for i in range(len(train_labels))]
        test_type = ["test" for i in range(len(test_labels))]
        types = train_type + test_type
        df_dict = {
            "labels": labels,
            "type": types
        }

        df = pd.DataFrame(df_dict, index=pd.Index(range(len(labels)), name="idx"))
        return df

    def sample_table_AL(self):
        """
        sample table for active learning scenarios
        :return:
        """
        df = self.sample_table()
        new_selected_epoch = [-1 for _ in range(len(self.training_labels)+len(self.testing_labels))]
        new_selected_epoch = np.array(new_selected_epoch)
        for epoch_id in range(self.data_provider.s, self.data_provider.e + 1, self.data_provider.p):
            labeled = np.array(self.get_epoch_index(epoch_id))
            new_selected_epoch[labeled] = epoch_id
        df["al_selected_epoch"] = new_selected_epoch.tolist()
        return df

    # def sample_table_noisy(self):
    #     df = self.sample_table()
    #     noisy_data = self.noisy_data_index()
    #     is_noisy = np.array([False for _ in range(len(self.training_labels)+len(self.testing_labels))])
    #     is_noisy[noisy_data] = True

    #     original_label = self.get_original_labels().tolist()
    #     test_labels = self.testing_labels.cpu().numpy().tolist()
    #     for ele in test_labels:
    #         original_label.append(ele)
    #     # original_label.extend(test_labels)

    #     df["original_label"] = original_label
    #     df["is_noisy"] = is_noisy.tolist()
    #     return df

    # customized features
    def filter_label(self, label):
        try:
            index = self.classes.index(label)
        except:
            index = -1
        train_labels = self.data_provider.train_labels()
        test_labels = self.data_provider.test_labels()
        labels = np.concatenate((train_labels, test_labels), 0)
        idxs = np.argwhere(labels == index)
        idxs = np.squeeze(idxs)
        return idxs

    def filter_type(self, type, epoch_id):
        if type == "train":
            res = self.get_epoch_index(epoch_id)
        elif type == "test":
            train_num = self.data_provider.train_num
            test_num = self.data_provider.test_num
            res = list(range(train_num, test_num, 1))
        elif type == "unlabel":
            labeled = np.array(self.get_epoch_index(epoch_id))
            train_num = self.data_provider.train_num
            all_data = np.arange(train_num)
            unlabeled = np.setdiff1d(all_data, labeled)
            res = unlabeled.tolist()
        # elif type == "noisy":
        #     res = self.noisy_data_index()
        else:
            # all data
            train_num = self.data_provider.train_num
            test_num = self.data_provider.test_num
            res = list(range(0, train_num + test_num, 1))
        return res

    def filter_prediction(self, pred):
        pass

    def get_epoch_index(self, epoch_id):
        """get the training data index for an epoch"""
        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        with open(index_file, 'r') as f:
            index = json.load(f)
        return index
