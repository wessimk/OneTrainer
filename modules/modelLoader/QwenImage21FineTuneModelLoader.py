from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.qwen.QwenImage21ModelLoader import QwenImage21ModelLoader
from modules.util.enum.ModelType import ModelType

QwenImage21FineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.QWEN_IMAGE_21: "resources/sd_model_spec/qwen-image-21.json"},
    model_class=QwenImage21Model,
    model_loader_class=QwenImage21ModelLoader,
    embedding_loader_class=None,
)
