from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ModelType import ModelType


class SampleFrameController:
    def __init__(
            self,
            sample: SampleConfig,
            model_type: ModelType,
            custom_conditioning_image: bool = False,
    ):
        self.sample = sample
        self.model_type = model_type
        self.custom_conditioning_image = custom_conditioning_image

    def is_flow_matching(self) -> bool:
        return self.model_type.is_flow_matching()

    def is_inpainting_model(self) -> bool:
        return self.model_type.has_mask_input()

    def supports_conditioning_image(self) -> bool:
        return self.custom_conditioning_image and self.model_type.is_qwen_image_21()

    def is_video_model(self) -> bool:
        return self.model_type.is_video_model()

    def supports_negative_prompt(self) -> bool:
        return self.model_type.supports_negative_prompt()
