from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ModelType import ModelType


class SampleParamsWindowController:
    def __init__(
            self,
            sample: SampleConfig,
            model_type: ModelType | None = None,
            custom_conditioning_image: bool = False,
    ):
        self.sample = sample
        self.model_type = model_type
        self.custom_conditioning_image = custom_conditioning_image
