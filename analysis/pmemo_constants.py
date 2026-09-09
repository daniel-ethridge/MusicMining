import os


class Paths:
    pmemo = "./datasets/PMEmo2019/PMEmo2019"
    pmemo_audio_dir = os.path.join(pmemo, "chorus")
    pmemo_static_annots_file = os.path.join(pmemo, "annotations/static_annotations.csv")
    pmemo_database = "./analysis/database/pmemo.db"


class SqlNames:
    annotations_table = "annotations"
    accompaniment_audio_table = "accompaniment_audio_features"
    vocals_audio_table = "vocal_audio_features"
    original_audio_table = "original_audio_features"
