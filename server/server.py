from msilib.schema import Error
from flask import request, Response, Flask, jsonify, make_response
from flask_cors import CORS, cross_origin

import os
import sys
import torch
import numpy as np
import json
import tensorflow as tf


# flask for API server
app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/updateProjection', methods=["POST", "GET"])
@cross_origin()
def update_projection():
    res = request.get_json()
    CONTENT_PATH = os.path.normpath(res['path'])
    EPOCH = int(res['iteration'])
    resolution = int(res['resolution'])
    predicates = res["predicates"]

    sys.path.append(content_path)
    try:
        from Model.model import resnet18
        net = resnet18()
    except:
        from Model.model import ResNet18
        net = ResNet18()

    # TODO fix
    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    DATASET = "cifar10"
    GPU_ID = 1

    LAMBDA = singleVis.config.dataset_config[DATASET]["LAMBDA"]
    EPOCH_START = singleVis.config.dataset_config[DATASET]["EPOCH_START"]
    EPOCH_END = singleVis.config.dataset_config[DATASET]["EPOCH_END"]
    EPOCH_PERIOD = singleVis.config.dataset_config[DATASET]["EPOCH_PERIOD"]

    # define hyperparameters
    DEVICE = torch.device("cuda:{:d}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

    content_path = CONTENT_PATH
    sys.path.append(content_path)

    data_provider = singleVis.data.DataProvider(content_path, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, verbose=1)
    model = singleVis.SingleVisualizationModel.SingleVisualizationModel(input_dims=512, output_dims=2, units=256)
    negative_sample_rate = 5
    min_dist = .1
    _a, _b = find_ab_params(1.0, min_dist)
    umap_loss_fn = singleVis.losses.UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
    recon_loss_fn = singleVis.losses.ReconstructionLoss(beta=1.0)
    criterion = singleVis.losses.SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

    trainer = singleVis.trainer.SingleVisTrainer(model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, edge_loader=None, DEVICE=DEVICE)
    trainer.load(file_path=os.path.join(data_provider.model_path,"tnn.pth"))
    trainer.model.eval()

    vis = singleVis.visualizer.visualizer(data_provider, trainer.model, resolution, 10, classes)
    evaluator = singleVis.eval.evaluator.Evaluator(data_provider, trainer)

    # load parameters

    train_data = data_provider.train_representation(EPOCH)
    test_data = data_provider.test_representation(EPOCH)
    all_data = np.concatenate((train_data, test_data), axis=0)

    embedding_2d = trainer.model.encoder(
        torch.from_numpy(all_data).to(dtype=torch.float32, device=trainer.DEVICE)).cpu().detach().numpy().tolist()

    train_labels = data_provider.train_labels(EPOCH)
    test_labels = data_provider.test_labels(EPOCH)
    labels = np.concatenate((train_labels, test_labels), axis=0).tolist()

    training_data_number = train_data.shape[0]
    testing_data_number = test_data.shape[0]
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))

    grid, decision_view = vis.get_epoch_decision_view(EPOCH, resolution)

    grid = grid.reshape((-1, 2)).tolist()
    decision_view = decision_view * 255
    decision_view = decision_view.reshape((-1, 3)).astype(int).tolist()

    color = vis.get_standard_classes_color() * 255
    color = color.astype(int).tolist()
    # TODO fix its structure
    evaluation = evaluator.get_eval()


from flask import request, Response, Flask, jsonify, make_response
from flask_cors import CORS, cross_origin

import os
import sys
import time
import json
import torch
import pandas as pd
import numpy as np
from umap.umap_ import find_ab_params
from sqlalchemy import create_engine, text

from antlr4 import *
from MyGrammar.MyGrammarLexer import MyGrammarLexer
from MyGrammar.MyGrammarParser import MyGrammarParser
from MyGrammarAdapter import MyGrammarPrintListener

from ..singleVis.SingleVisualizationModel import SingleVisualizationModel
from ..singleVis.data import DataProvider
from ..singleVis.eval.evaluator import Evaluator
from ..singleVis.trainer import SingleVisTrainer
from ..singleVis.losses import ReconstructionLoss, UmapLoss, SingleVisLoss
from ..singleVis.visualizer import visualizer
from BackendAdapter import TimeVisBackend


# flask for API server
app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'


def record(string):
    with open("record.txt", "a") as file_object:
        file_object.write(string+"\n")


@app.route('/load', methods=["POST", "GET"])
@cross_origin()
def load():
    res = request.get_json()
    CONTENT_PATH = os.path.normpath(res['path'])
    sys.path.append(CONTENT_PATH)

    # load hyperparameters
    config_file = os.path.join(CONTENT_PATH, "config.json")
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except:
        raise Error("config file not exists...")

    CLASSES = config["CLASSES"]
    DATASET = config["DATASET"]
    LAMBDA = config["TRAINING"]["LAMBDA"]
    EPOCH_START = config["EPOCH_START"]
    EPOCH_END = config["EPOCH_END"]
    EPOCH_PERIOD = config["EPOCH_PERIOD"]
    SUBJECT_MODEL_NAME = config["TRAINING"]["SUBJECT_MODEL_NAME"]
    VIS_MODEL_NAME = config["VISUALIZATION"]["VIS_MODEL_NAME"]
    RESOLUTION = config["VISUALIZATION"]["RESOLUTION"]


    # define hyperparameters
    DEVICE = torch.device("cuda:0"0 if torch.cuda.is_available() else "cpu")

    import Model.model as subject_model
    try:
        net = eval("subject_model.{}()".format(SUBJECT_MODEL_NAME))
    except:
        raise Error("No subject model found in model.py...")

    data_provider = DataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, verbose=1)
    model = SingleVisualizationModel.SingleVisualizationModel(input_dims=512, output_dims=2, units=256)
    negative_sample_rate = 5
    min_dist = .1
    _a, _b = find_ab_params(1.0, min_dist)
    umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
    recon_loss_fn = ReconstructionLoss(beta=1.0)
    criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

    trainer = SingleVisTrainer(model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, edge_loader=None, DEVICE=DEVICE)
    trainer.load(file_path=os.path.join(data_provider.model_path,"{}".format(VIS_MODEL_NAME)))
    trainer.model.eval()

    vis = visualizer(data_provider, trainer.model, RESOLUTION, 10, CLASSES)
    evaluator = Evaluator(data_provider, trainer)
    timevis = TimeVisBackend(data_provider, trainer, evaluator)


    sql_engine       = create_engine('mysql+pymysql://xg:password@localhost/dviDB', pool_recycle=3600)
    db_connection    = sql_engine.connect()

    # Search the following tables in MYSQL database and drop them if they exist
    sql_engine.execute(text('DROP TABLE IF EXISTS SubjectModel;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS VisModel;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS Sample;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS NoisySample;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS AlSample;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS PredSample;'))

    # Create the SubjectModel table in MYSQL database and insert the data
    table_subject_model = "SubjectModel"
    data_subject_model = timevis.subject_model_table()
    data_subject_model.to_sql(table_subject_model, db_connection, if_exists='fail');

    # Create the VisModel table in MYSQL database and insert the data
    table_vis_model = "VisModel"
    data_vis_model = timevis.vis_model_table()
    data_vis_model.to_sql(table_vis_model, db_connection, if_exists='fail');

    # Create the Sample table in MYSQL database and insert the data
    table_sample = "Sample"
    data_sample = timevis.sample_table()
    data_sample.to_sql(table_sample, db_connection, if_exists='fail');

    # For nosiy or active learning data, currently not tested yet
    if "noisy" in CONTENT_PATH:     
        table_noisy_sample = "NoisySample"
        data_noisy_sample = timevis.sample_table_noisy()
        data_noisy_sample.to_sql(table_noisy_sample, db_connection, if_exists='fail');
    elif "active" in CONTENT_PATH:
        table_al_sample = "AlSample"
        data_al_sample = timevis.sample_table_AL()
        data_al_sample.to_sql(table_al_sample, db_connection, if_exists='fail');
    
    # Ablation starts here
    # Store prediction, deltaboundary true/false for all samples in all epochs in PredSample table
    all_prediction_list = []
    all_deltab_list = []
    all_epochs_list = []
    all_idx_list = []
    for iteration in range(data_provider.s, data_provider.e+1, data_provider.p):
        print("iteration", iteration)
        train_data = data_provider.train_representation(iteration)
        test_data = data_provider.test_representation(iteration)
        all_data = np.concatenate((train_data, test_data), axis=0)

        prediction = data_provider.get_pred(iteration, all_data).argmax(-1)
        deltab = data_provider.is_deltaB(iteration, all_data)

        count = 0
        for idx,_ in enumerate(prediction):
            all_prediction_list.append(prediction[idx])
            all_deltab_list.append(deltab[idx])
            all_epochs_list.append(iteration)
            all_idx_list.append(count)
            count += 1

    data_pred_sample = pd.DataFrame(list(zip(all_idx_list, all_epochs_list, all_prediction_list, all_deltab_list)),
               columns =['idx', 'epoch', 'pred', 'deltab'])
    table_pred_sample = "PredSample"
    data_pred_sample.to_sql(table_pred_sample, db_connection, if_exists='fail')

    db_connection.close()


@app.route('/updateProjection', methods=["POST", "GET"])
@cross_origin()
def update_projection():
    res = request.get_json()
    content_path = os.path.normpath(res['path'])
    resolution = int(res['resolution'])
    predicates = res["predicates"]
    sys.path.append(content_path)

    try:
        from Model.model import ResNet18
        net = ResNet18()
    except:
        from Model.model import resnet18
        net = resnet18()

    # Retrieving hyperparameters from json file to be passed as  parameters for MMS model
    with open(content_path+"/config.json") as file:
        data = json.load(file)
        for key in data:
            if key=="dataset":
                dataset = data[key]
            elif key=="epoch_start":
                start_epoch = int(data[key])
            elif key=="epoch_end":
                end_epoch = int(data[key])
            elif key=="epoch_period":
                period = int(data[key])
            elif key=="embedding_dim":
                embedding_dim = int(data[key])
            elif key=="num_classes":
                num_classes = int(data[key])
            elif key=="classes":
                classes = range(num_classes)
            elif key=="temperature":
                temperature = float(data[key])
            elif key=="attention":
                if int(data[key]) == 0:
                    attention = False
                else:
                    attention = True
            elif key=="cmap":
                cmap = data[key]
            elif key=="resolution":
                resolution = int(data[key])
            elif key=="temporal":
                if int(data[key]) == 0:
                    temporal = False
                else:
                    temporal = True
            elif key=="transfer_learning":
                transfer_learning = int(data[key])
            elif key=="step3":
                step3 = int(data[key])
            elif key=="split":
                split = int(data[key])
            elif key=="advance_border_gen":
                if int(data[key]) == 0:
                    advance_border_gen = False
                else:
                    advance_border_gen = True
            elif key=="alpha":
                alpha = float(data[key])
            elif key=="withoutB":
                if int(data[key]) == 0:
                    withoutB = False
                else:
                    withoutB = True
            elif key=="attack_device":
                attack_device = data[key]

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    mms = MMS(content_path, net, start_epoch, end_epoch, period, embedding_dim, num_classes, classes, temperature=temperature, cmap=cmap, resolution=resolution, verbose=1, attention=attention, 
              temporal=False, split=split, alpha=alpha, advance_border_gen=advance_border_gen, withoutB=withoutB, attack_device="cpu")
    iteration = int(res['iteration'])*period


    train_data = mms.get_data_pool_repr(iteration)
    # train_data = mms.get_epoch_train_repr_data(iteration)
    test_data = mms.get_epoch_test_repr_data(iteration)
    all_data = np.concatenate((train_data, test_data), axis=0)

    embedding_2d = mms.batch_project(all_data, iteration).tolist()
    train_labels = mms.training_labels.cpu().numpy()
    test_labels = mms.testing_labels.cpu().numpy()
    labels = np.concatenate((train_labels, test_labels),axis=0).tolist()

    training_data_number = train_data.shape[0]
    testing_data_number = test_data.shape[0]
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))

    grid, decision_view = mms.get_epoch_decision_view(iteration, resolution)

    grid = grid.reshape((-1, 2)).tolist()
    decision_view = decision_view * 255
    decision_view = decision_view.reshape((-1, 3)).astype(int).tolist()

    color = mms.get_standard_classes_color() * 255
    color = color.astype(int).tolist()

    evaluation = mms.get_eval(iteration)

    label_color_list = []
    label_list = []
    for label in labels:
        label_color_list.append(color[int(label)])
        label_list.append(classes[int(label)])

    prediction_list = []
    prediction = mms.get_pred(iteration, all_data).argmax(-1)
    # classes_map = dict()
    # for i in range(10):
    #     classes_map[i] = classes[i]
    # prediction_list = np.vectorize(classes_map.get)(prediction).tolist()
    for pred in prediction:
        prediction_list.append(classes[pred])

    max_iter = 0
    path_files = os.listdir(mms.model_path)
    for file in path_files:
        if "Epoch" in file:
            max_iter += 1
        if "Epoch_0" in file:
            max_iter -= 1

    _, conf_diff = mms.batch_inv_preserve(iteration, all_data)
    current_index = mms.get_epoch_index(iteration)

    new_index = mms.get_new_index(iteration)

    noisy_data = mms.noisy_data_index()

    original_labels = mms.get_original_labels()
    original_label_list = []
    for label in original_labels:
        original_label_list.append(classes[label])

    uncertainty_diversity_tot_dict = {}
    uncertainty_diversity_tot_dict['uncertainty'] = mms.get_uncertainty_score(iteration)
    uncertainty_diversity_tot_dict['diversity'] = mms.get_diversity_score(iteration)
    uncertainty_diversity_tot_dict['tot'] = mms.get_total_score(iteration)

    uncertainty_ranking_list = [i[0] for i in sorted(enumerate(uncertainty_diversity_tot_dict['uncertainty']), key=lambda x: x[1])]
    diversity_ranking_list = [i[0] for i in sorted(enumerate(uncertainty_diversity_tot_dict['diversity']), key=lambda x: x[1])]
    tot_ranking_list = [i[0] for i in sorted(enumerate(uncertainty_diversity_tot_dict['tot']), key=lambda x: x[1])]
    uncertainty_diversity_tot_dict['uncertainty_ranking'] = uncertainty_ranking_list
    uncertainty_diversity_tot_dict['diversity_ranking'] = diversity_ranking_list
    uncertainty_diversity_tot_dict['tot_ranking'] = tot_ranking_list

    selected_points = np.arange(mms.get_dataset_length())
    for key in predicates.keys():
        if key == "new_selection":
            tmp = np.array(mms.get_new_index(int(predicates[key])))
        elif key == "label":
            tmp = np.array(mms.filter_label(predicates[key]))
        elif key == "type":
            tmp = np.array(mms.filter_type(predicates[key], int(iteration)))
        else:
            tmp = np.arange(mms.get_dataset_length())
        selected_points = np.intersect1d(selected_points, tmp)

    sys.path.remove(content_path)


    return make_response(jsonify({'result': embedding_2d, 'grid_index': grid, 'grid_color': decision_view,
                                  'label_color_list': label_color_list, 'label_list': label_list,
                                  'maximum_iteration': max_iter, 'training_data': current_index,
                                  'testing_data': testing_data_index, 'evaluation': evaluation,
                                  'prediction_list': prediction_list, 'new_selection': new_index,
                                  'noisy_data': noisy_data, 'original_label_list': original_label_list,
                                  'inv_acc_list': conf_diff.tolist(),
                                  'uncertainty_diversity_tot': uncertainty_diversity_tot_dict,
                                  "selectedPoints":selected_points.tolist()}), 200)

@app.route('/query', methods=["POST"])
@cross_origin()
def filter():
    res = request.get_json()
    string = res["predicates"]["label"]
    content_path = os.path.normpath(res['content_path'])

    data =  InputStream(string)
    # lexer
    lexer = MyGrammarLexer(data)
    stream = CommonTokenStream(lexer)
    # parser
    parser = MyGrammarParser(stream)
    tree = parser.expr()
    # Currently this is hardcoded for CIFAR10, changes need to be made in future
    # Error will appear based on some of the queries sent
    model_epochs = [40, 80, 120, 160, 200]
    # evaluator
    listener = MyGrammarPrintListener(model_epochs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    statement = listener.result

    sql_engine       = create_engine('mysql+pymysql://xg:password@localhost/dviDB', pool_recycle=3600)
    db_connection    = sql_engine.connect()
    frame           = pd.read_sql(statement, db_connection);
    pd.set_option('display.expand_frame_repr', False)
    db_connection.close()
    result = []
    for _, row in frame.iterrows():
        for col in frame.columns:
            result.append(int(row[col]))
    return make_response(jsonify({"selectedPoints":result}), 200)


@app.route('/saveDVIselections', methods=["POST"])
@cross_origin()
def save_DVI_selections():
    data = request.get_json()
    indices = data["newIndices"]

    content_path = os.path.normpath(data['content_path'])
    iteration = data["iteration"]
    sys.path.append(content_path)
    try:
        from Model.model import ResNet18
        net = ResNet18()
    except:
        from Model.model import resnet18
        net = resnet18()

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    mms = MMS(content_path, net, 1, 20, 1, 512, 10, classes, cmap="tab10", neurons=256, verbose=1,
              temporal=False, split=-1, advance_border_gen=True, attack_device="cpu")
    mms.save_DVI_selection(iteration, indices)

    sys.path.remove(content_path)

    return make_response(jsonify({"message":"Save DVI selection succefully!"}), 200)

@app.route('/sprite', methods=["POST","GET"])
@cross_origin()
def sprite_image():
    path= request.args.get("path")
    sprite = tf.io.gfile.GFile(path, "rb")
    encoded_image_string = sprite.read()
    sprite.close()
    image_type = "image/png"
    return Response(encoded_image_string, status=200, mimetype=image_type)

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    with open('../tensorboard/tensorboard/plugins/projector/vz_projector/standalone_projector_config.json', 'r') as f:
        config = json.load(f)
        ip_adress = config["DVIServerIP"]
        port = config["DVIServerPort"]
    app.run(host=ip_adress, port=int(port))
