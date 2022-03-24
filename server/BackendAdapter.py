'''This class serves as a intermediate layer for tensorboard frontend and timeVis backend'''
import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax

# sys.path.append("..")
from singleVis.utils import *


class TimeVisBackend:
    def __init__(self, data_provider, trainer, evaluator) -> None:
        self.data_provider = data_provider
        self.trainer = trainer
        self.trainer.model.eval()
        self.evaluator = evaluator
    
    #################################################################################################################
    #                                                                                                               #
    #                                                data Panel                                                     #
    #                                                                                                               #
    #################################################################################################################

    def batch_project(self, data):
        embedding = self.trainer.model.encoder(torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return embedding
    
    def individual_project(self, data):
        embedding = self.trainer.model.encoder(torch.from_numpy(np.expand_dims(data, axis=0)).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return embedding.squeeze(axis=0)
    
    def batch_inverse(self, embedding):
        data = self.trainer.model.decoder(torch.from_numpy(embedding).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return data
    
    def individual_inverse(self, embedding):
        data = self.trainer.model.decoder(torch.from_numpy(np.expand_dims(embedding, axis=0)).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        return data.squeeze(axis=0)

    def batch_inv_preserve(self, epoch, data):
        """
        get inverse confidence for a single point
        :param epoch: int
        :param data: numpy.ndarray
        :return l: boolean, whether reconstruction data have the same prediction
        :return conf_diff: float, (0, 1), confidence difference
        """
        self.trainer.model.eval()
        embedding = self.batch_project(data)
        recon = self.batch_inverse(embedding)
    
        ori_pred = self.data_provider.get_pred(epoch, data)
        new_pred = self.data_provider.get_pred(epoch, recon)
        ori_pred = softmax(ori_pred, axis=1)
        new_pred = softmax(new_pred, axis=1)

        old_label = ori_pred.argmax(-1)
        new_label = new_pred.argmax(-1)
        l = old_label == new_label

        old_conf = [ori_pred[i, old_label[i]] for i in range(len(old_label))]
        new_conf = [new_pred[i, old_label[i]] for i in range(len(old_label))]
        old_conf = np.array(old_conf)
        new_conf = np.array(new_conf)

        conf_diff = old_conf - new_conf
        return l, conf_diff
    
    #################################################################################################################
    #                                                                                                               #
    #                                                Search Panel                                                   #
    #                                                                                                               #
    #################################################################################################################


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
    
    # TODO have not define noisy dataset yet

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



    #################################################################################################################
    #                                                                                                               #
    #                                             Helper Functions                                                  #
    #                                                                                                               #
    #################################################################################################################

    def get_epoch_index(self, epoch_id):
        """get the training data index for an epoch"""
        index_file = os.path.join(self.data_provider.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        index = load_labelled_data_index(index_file)
        return index
    
    #################################################################################################################
    #                                                                                                               #
    #                                          Case Studies Related                                                 #
    #                                                                                                               #
    #################################################################################################################     
    '''active learning'''
    def get_new_index(self, epoch):
        """get the index of new selection"""
        new_epoch = epoch + self.data_provider.p
        if new_epoch > self.data_provider.e:
            return list()

        index_file = os.path.join(self.data_provider.model_path, "Epoch_{:d}".format(epoch), "index.json")
        index = load_labelled_data_index(index_file)
        new_index_file = os.path.join(self.data_provider.model_path, "Epoch_{:d}".format(new_epoch), "index.json")
        new_index = load_labelled_data_index(new_index_file)

        idxs = []
        for i in new_index:
            if i not in index:
                idxs.append(i)

        return idxs

    '''Noise data(Mislabelled data)'''
    def noisy_data_index(self):
        """get noise data index"""
        index_file = os.path.join(self.data_provider.content_path, "index.json")
        if not os.path.exists(index_file):
            return list()
        return load_labelled_data_index(index_file)

    def get_original_labels(self):
        """
        get original dataset label(without noise)
        :return labels: list, shape(N,)
        """
        index_file = os.path.join(self.data_provider.content_path, "index.json")
        if not os.path.exists(index_file):
            return list()
        index = load_labelled_data_index(index_file)
        old_file = os.path.join(self.data_provider.content_path, "old_labels.json")
        old_labels = load_labelled_data_index(old_file)

        labels = np.copy(self.training_labels.cpu().numpy())
        labels[index] = old_labels

        return labels

    def get_uncertainty_score(self, epoch_id):
        try:
            uncertainty_score_path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch_id),"train_uncertainty_score.json")
            with open(uncertainty_score_path, "r") as f:
                train_uncertainty_score = json.load(f)

            uncertainty_score_path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch_id),"test_uncertainty_score.json")
            with open(uncertainty_score_path, "r") as f:
                test_uncertainty_score = json.load(f)

            uncertainty_score = train_uncertainty_score + test_uncertainty_score
            return uncertainty_score
        except FileNotFoundError:
            train_num = self.data_provider.train_num
            test_num = self.data_provider.test_num
            return [-1 for i in range(train_num+test_num)]

    def get_diversity_score(self, epoch_id):
        try:
            dis_score_path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch_id), "train_dis_score.json")
            with open(dis_score_path, "r") as f:
                train_dis_score = json.load(f)

            dis_score_path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch_id), "test_dis_score.json")
            with open(dis_score_path, "r") as f:
                test_dis_score = json.load(f)

            dis_score = train_dis_score + test_dis_score

            return dis_score
        except FileNotFoundError:
            train_num = self.data_provider.train_num
            test_num = self.data_provider.test_num
            return [-1 for i in range(train_num+test_num)]

    def get_total_score(self, epoch_id):
        try:
            total_score_path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch_id), "train_total_score.json")
            with open(total_score_path, "r") as f:
                train_total_score = json.load(f)

            total_score_path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch_id), "test_total_score.json")
            with open(total_score_path, "r") as f:
                test_total_score = json.load(f)

            total_score = train_total_score + test_total_score

            return total_score
        except FileNotFoundError:
            train_num = self.data_provider.train_num
            test_num = self.data_provider.test_num
            return [-1 for i in range(train_num+test_num)]

    def save_DVI_selection(self, epoch_id, indices):
        """
        save the selected index message from DVI frontend
        :param epoch_id:
        :param indices: list, selected indices
        :return:
        """
        save_location = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch_id), "DVISelection.json")
        with open(save_location, "w") as f:
            json.dump(indices, f)
