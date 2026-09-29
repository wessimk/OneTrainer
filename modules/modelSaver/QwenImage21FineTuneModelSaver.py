from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.qwen.QwenImage21ModelSaver import QwenImage21ModelSaver
from modules.util.enum.ModelType import ModelType

QwenImage21FineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.QWEN_IMAGE_21,
    model_class=QwenImage21Model,
    model_saver_class=QwenImage21ModelSaver,
    embedding_saver_class=None,
)
