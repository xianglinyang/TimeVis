from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def draw_trajectory(embeddings):
    fig = plt.figure()
    plt.quiver(embeddings[:-1, 0], embeddings[:-1, 1], embeddings[1:, 0]-embeddings[:-1, 0], embeddings[1:, 1]-embeddings[:-1, 1],scale_units='xy', angles='xy', scale=1)  
    x_min = embeddings[:, 0].min()
    x_max = embeddings[:, 0].max()
    y_min = embeddings[:, 1].min()
    y_max = embeddings[:, 1].max()

    plt.xlim((x_min-0.1*(x_max-x_min), x_max+0.1*(x_max-x_min)))
    plt.ylim((y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min)))
    plt.savefig("/home/xianglin/projects/git_space/SingleVisualization/singleVis/trac.png")

def draw_two_trajectories(embeddings1, embeddings2):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.quiver(embeddings1[:-1, 0], embeddings1[:-1, 1], embeddings1[1:, 0]-embeddings1[:-1, 0], embeddings1[1:, 1]-embeddings1[:-1, 1],scale_units='xy', angles='xy', scale=1)  
    x_min = embeddings1[:, 0].min()
    x_max = embeddings1[:, 0].max()
    y_min = embeddings1[:, 1].min()
    y_max = embeddings1[:, 1].max()

    ax.set_xlim((x_min-0.1*(x_max-x_min), x_max+0.1*(x_max-x_min)))
    ax.set_ylim((y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min)))
    ax.set_title("DVI")

    ax = fig.add_subplot(122)
    ax.quiver(embeddings2[:-1, 0], embeddings2[:-1, 1], embeddings2[1:, 0]-embeddings2[:-1, 0], embeddings2[1:, 1]-embeddings2[:-1, 1],scale_units='xy', angles='xy', scale=1)  
    x_min = embeddings2[:, 0].min()
    x_max = embeddings2[:, 0].max()
    y_min = embeddings2[:, 1].min()
    y_max = embeddings2[:, 1].max()

    ax.set_xlim((x_min-0.1*(x_max-x_min), x_max+0.1*(x_max-x_min)))
    ax.set_ylim((y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min)))
    ax.set_title("DVI-T")

    plt.savefig("/home/xianglin/projects/git_space/SingleVisualization/singleVis/trac.png")
    

if __name__ == "__main__":
    embeddings_x = np.arange(5)
    embeddings_y = np.arange(2,7,1)
    embeddings_z = np.arange(-2, -7, -1)
    embeddings1 = np.concatenate((embeddings_x[None,], embeddings_y[None,]), axis=0)
    embeddings1 = embeddings1.transpose()
    embeddings2 = np.concatenate((embeddings_x[None,], embeddings_z[None,]), axis=0)
    embeddings2 = embeddings2.transpose()
    # draw_trajectory(embeddings)

    draw_two_trajectories(embeddings1, embeddings2)
    


