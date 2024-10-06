import os
import numpy as np
import random
import torch
from datetime import datetime
from torch.utils.data import Dataset
from enum import Enum, auto


class ModelName(Enum):
    airplane = auto()
    bag = auto()
    basket = auto()
    bathtub = auto()
    bed = auto()
    bench = auto()
    bottle = auto()
    bowl = auto()
    bus = auto()
    cabinet = auto()
    can = auto()
    camera = auto()
    cap = auto()
    car = auto()
    chair = auto()
    clock = auto()
    dishwasher = auto()
    monitor = auto()
    table = auto()
    telephone = auto()
    tin_can = auto()
    tower = auto()
    train = auto()
    keyboard = auto()
    earphone = auto()
    faucet = auto()
    file = auto()
    guitar = auto()
    helmet = auto()
    jar = auto()
    knife = auto()
    lamp = auto()
    laptop = auto()
    speaker = auto()
    mailbox = auto()
    microphone = auto()
    microwave = auto()
    motorcycle = auto()
    mug = auto()
    piano = auto()
    pillow = auto()
    pistol = auto()
    pot = auto()
    printer = auto()
    remote_control = auto()
    rifle = auto()
    rocket = auto()
    skateboard = auto()
    sofa = auto()
    stove = auto()
    vessel = auto()
    washer = auto()
    cellphone = auto()
    birdhouse = auto()
    bookshelf = auto()


path_table = {
    ModelName.airplane: "02691156",
    ModelName.bag: "02773838",
    ModelName.basket: "02801938",
    ModelName.bathtub: "02808440",
    ModelName.bed: "02818832",
    ModelName.bench: "02828884",
    ModelName.bottle: "02876657",
    ModelName.bowl: "02880940",
    ModelName.bus: "02924116",
    ModelName.cabinet: "02933112",
    ModelName.can: "02747177",
    ModelName.camera: "02942699",
    ModelName.cap: "02954340",
    ModelName.car: "02958343",
    ModelName.chair: "03001627",
    ModelName.clock: "03046257",
    ModelName.dishwasher: "03207941",
    ModelName.monitor: "03211117",
    ModelName.table: "04379243",
    ModelName.telephone: "04401088",
    ModelName.tin_can: "02946921",
    ModelName.tower: "04460130",
    ModelName.train: "04468005",
    ModelName.keyboard: "03085013",
    ModelName.earphone: "03261776",
    ModelName.faucet: "03325088",
    ModelName.file: "03337140",
    ModelName.guitar: "03467517",
    ModelName.helmet: "03513137",
    ModelName.jar: "03593526",
    ModelName.knife: "03624134",
    ModelName.lamp: "03636649",
    ModelName.laptop: "03642806",
    ModelName.speaker: "03691459",
    ModelName.mailbox: "03710193",
    ModelName.microphone: "03759954",
    ModelName.microwave: "03761084",
    ModelName.motorcycle: "03790512",
    ModelName.mug: "03797390",
    ModelName.piano: "03928116",
    ModelName.pillow: "03938244",
    ModelName.pistol: "03948459",
    ModelName.pot: "03991062",
    ModelName.printer: "04004475",
    ModelName.remote_control: "04074963",
    ModelName.rifle: "04090263",
    ModelName.rocket: "04099429",
    ModelName.skateboard: "04225987",
    ModelName.sofa: "04256520",
    ModelName.stove: "04330267",
    ModelName.vessel: "04530566",
    ModelName.washer: "04554684",
    ModelName.cellphone: "02992529",
    ModelName.birdhouse: "02843684",
    ModelName.bookshelf: "02871439",
}


def get_all_path(root_dir, model_name, is_train):
    all_path = []
    middle_folder = "train" if is_train else "val"
    path = os.path.join(root_dir, path_table[model_name], middle_folder)
    if not os.path.isdir(path):
        raise Exception(path + " is not a folder path")
    for file in os.listdir(path):
        if not file.endswith(".npy"):
            continue
        all_path.append(os.path.join(path, file))
    return all_path


class PointflowDataLoader(Dataset):
    def __init__(self, root_dir, model_name, is_train):
        self.root_dir = root_dir
        self.model_name = model_name
        self.is_train = is_train
        self.all_points = []
        all_path = get_all_path(root_dir, model_name, is_train)

        np.random.seed(datetime.now().second + datetime.now().microsecond)
        np.random.shuffle(all_path)

        for file in all_path:
            try:
                data = np.load(file)  # (15000,3)
            except Exception as e:
                print(e + 'file: ' + file)
                continue
            self.all_points.append(data[np.newaxis, :])
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)

        # Normalization
        dim = self.all_points[2]
        self.all_points_mean = (
            self.all_points.reshape(-1, dim).mean(axis=0).reshape(1, 1, dim)
        )
        self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

    def __getitem__(self, idx):
        target_points = self.all_points[idx]
        eval_cloud = target_points[1::2].copy().T
        cloud = target_points[::2].T

        return {"idx": idx, "cloud": cloud, "eval_cloud": eval_cloud}
