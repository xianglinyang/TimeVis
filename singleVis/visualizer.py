import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from torch._C import device

class visualizer:
    def __init__(self, data_provider, sv_model, resolution, class_num, classes, cmap='tab10'):
        self.data_provider = data_provider
        self.model = sv_model
        self.class_num = class_num
        self.cmap = plt.get_cmap(cmap)
        self.classes = classes
        self.resolution= resolution

        self.model.eval()
    
    def _init_plot(self):
        '''
        Initialises matplotlib artists and plots. from DeepView and DVI
        '''
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self.ax.set_title("DVI-T visualization")
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        # labels = prediction
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '.', label=self.classes[c], ms=1,
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        # labels != prediction, labels be a large circle
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                fillstyle='full', ms=3, mew=2.5, zorder=3)
            self.sample_plots.append(plot[0])

        # labels != prediction, prediction stays inside of circle
        for c in range(self.class_num):
            color = self.cmap(c / (self.class_num - 1))
            plot = self.ax.plot([], [], '.', markeredgecolor=color,
                                fillstyle='full', ms=2, zorder=4)
            self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        # self.fig.canvas.mpl_connect('pick_event', self.show_sample)
        # self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False
        self.ax.legend()
    
    def get_epoch_plot_measures(self, epoch):
        """get plot measure for visualization"""
        data = self.data_provider.train_representation(epoch)
        data = torch.from_numpy(data).to(device=self.data_provider.DEVICE, dtype=torch.float)
        embedded = self.model.encoder(data).cpu().detach().numpy()

        ebd_min = np.min(embedded, axis=0)
        ebd_max = np.max(embedded, axis=0)
        ebd_extent = ebd_max - ebd_min

        x_min, y_min = ebd_min - 0.1 * ebd_extent
        x_max, y_max = ebd_max + 0.1 * ebd_extent

        x_min = min(x_min, y_min)
        y_min = min(x_min, y_min)
        x_max = max(x_max, y_max)
        y_max = max(x_max, y_max)

        return x_min, y_min, x_max, y_max
    
    def get_epoch_decision_view(self, epoch, resolution):
        '''
        get background classifier view
        :param epoch_id: epoch that need to be visualized
        :param resolution: background resolution
        :return:
            grid_view : numpy.ndarray, self.resolution,self.resolution, 2
            decision_view : numpy.ndarray, self.resolution,self.resolution, 3
        '''
        print('Computing decision regions ...')

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        # create grid
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(grid.reshape(grid.shape[0], -1), 0, 1)

        # map gridmpoint to images
        grid = torch.from_numpy(grid).to(device=self.data_provider.DEVICE, dtype=torch.float)
        grid_samples = self.model.decoder(grid).cpu().detach().numpy()

        mesh_preds = self.data_provider.get_pred(epoch, grid_samples)
        mesh_preds = mesh_preds + 1e-8

        sort_preds = np.sort(mesh_preds)
        diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
        border = np.zeros(len(diff), dtype=np.uint8) + 0.05
        border[diff < 0.15] = 1
        diff[border == 1] = 0.

        diff = diff/diff.max()
        diff = diff*0.9

        mesh_classes = mesh_preds.argmax(axis=1)
        mesh_max_class = max(mesh_classes)
        color = self.cmap(mesh_classes / mesh_max_class)

        diff = diff.reshape(-1, 1)

        color = color[:, 0:3]
        # color = diff * 0.5 * color + (1 - diff) * np.ones(color.shape, dtype=np.uint8)
        color = diff * 0.5 * color + (1 - diff) * np.ones(color.shape, dtype=np.uint8)
        decision_view = color.reshape(resolution, resolution, 3)
        grid_view = grid.reshape(resolution, resolution, 2)
        return grid_view, decision_view
    
    def savefig(self, epoch, path="vis"):
        '''
        Shows the current plot.
        '''
        self._init_plot()

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'res: %d'
        desc = params_str % (self.resolution)
        self.desc.set_text(desc)

        train_data = self.data_provider.train_representation(epoch)
        train_labels = self.data_provider.train_labels(epoch)
        pred = self.data_provider.get_pred(epoch, train_data)
        pred = pred.argmax(axis=1)

        train_data = torch.from_numpy(train_data).to(device=self.data_provider.DEVICE, dtype=torch.float)
        embedding = self.model.encoder(train_data).cpu().detach().numpy()
        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels == pred)]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels != pred)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())
        #
        for c in range(self.class_num):
            data = embedding[np.logical_and(pred == c, train_labels != pred)]
            self.sample_plots[2*self.class_num + c].set_data(data.transpose())

        # if os.name == 'posix':
        #     self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # plt.text(-8, 8, "test", fontsize=18, style='oblique', ha='center', va='top', wrap=True)
        plt.savefig(path)
    
    def savefig_cus(self, epoch, data, pred, labels, path="vis"):
        '''
        Shows the current plot with given data
        '''
        self._init_plot()

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'res: %d'
        desc = params_str % (self.resolution)
        self.desc.set_text(desc)

        # train_data = self.data_provider.train_representation(epoch)
        # train_labels = self.data_provider.train_labels(epoch)
        # pred = self.data_provider.get_pred(epoch, train_data)
        # pred = pred.argmax(axis=1)

        data = torch.from_numpy(data).to(device=self.data_provider.DEVICE, dtype=torch.float)
        embedding = self.model.encoder(data).cpu().detach().numpy()
        for c in range(self.class_num):
            data = embedding[np.logical_and(labels == c, labels == pred)]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(labels == c, labels != pred)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())
        #
        for c in range(self.class_num):
            data = embedding[np.logical_and(pred == c, labels != pred)]
            self.sample_plots[2*self.class_num + c].set_data(data.transpose())

        # if os.name == 'posix':
        #     self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # plt.text(-8, 8, "test", fontsize=18, style='oblique', ha='center', va='top', wrap=True)
        plt.savefig(path)