from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.qwen.QwenLoRASaver import QwenLoRASaver
from modules.util.enum.ModelType import ModelType

QwenImage21LoRAModelSaver = make_lora_model_saver(
    ModelType.QWEN_IMAGE_21,
    model_class=QwenImage21Model,
    lora_saver_class=QwenLoRASaver,
    embedding_saver_class=None,
)
