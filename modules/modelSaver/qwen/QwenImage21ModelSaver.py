import os
from pathlib import Path

from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

from safetensors.torch import save_file


class QwenImage21ModelSaver(DtypeModelSaverMixin):
    def __save_diffusers(self, model, destination, dtype):
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        save_pipeline = self._copy_pipeline_to_dtype(pipeline, dtype, pipeline.processor)
        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

    def __save_transformer(self, model, destination, dtype):
        state_dict = self._convert_state_dict_dtype(model.transformer.state_dict(), dtype)
        self._convert_state_dict_to_contiguous(state_dict)
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(state_dict, destination, self._create_safetensors_header(model, state_dict))

    def save(self, model: QwenImage21Model, output_model_format, output_model_destination, dtype):
        match output_model_format:
            case ModelFormat.DIFFUSERS | ModelFormat.INTERNAL:
                self.__save_diffusers(model, output_model_destination, dtype if output_model_format != ModelFormat.INTERNAL else None)
            case ModelFormat.LEGACY_SAFETENSORS | ModelFormat.ORIGINAL_TRANSFORMER:
                self.__save_transformer(model, output_model_destination, dtype)
            case _:
                raise NotImplementedError(f"Unsupported output format: {output_model_format}")
