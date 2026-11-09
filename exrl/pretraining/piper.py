"""Piper observation and action transforms shared by fine-tuning and inference."""

import dataclasses

import numpy as np
from openpi import transforms as t
from openpi.training import config as c


@dataclasses.dataclass(frozen=True)
class PiperInputs(t.DataTransformFn):
    def __call__(self, data):
        result = dict(
            state=np.asarray(data["observation/state"], dtype=np.float32),
            image=dict(
                base_0_rgb=data["observation/image"],
                left_wrist_0_rgb=data["observation/wrist_image"],
                right_wrist_0_rgb=np.zeros_like(data["observation/image"]),
            ),
            image_mask=dict(
                base_0_rgb=np.True_, left_wrist_0_rgb=np.True_, right_wrist_0_rgb=np.False_
            ),
        )
        for key in ("actions", "prompt"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs(t.DataTransformFn):
    def __call__(self, data):
        return dict(actions=np.asarray(data["actions"][:, :7], dtype=np.float32))


@dataclasses.dataclass(frozen=True)
class PiperData(c.DataConfigFactory):
    prompt: str = ""
    action_mode: str = "state_delta"

    def create(self, assets_dirs, model_config):
        if self.action_mode not in ("state_delta", "absolute"):
            raise ValueError("Choose state_delta or absolute joint actions")
        repack = t.Group(
            inputs=[
                t.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        group = t.Group(inputs=[PiperInputs()], outputs=[PiperOutputs()])
        if self.action_mode == "state_delta":
            mask = t.make_bool_mask(6, -1)
            group = group.push(inputs=[t.DeltaActions(mask)], outputs=[t.AbsoluteActions(mask)])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=group,
            model_transforms=c.ModelTransformFactory(default_prompt=self.prompt)(model_config),
            action_sequence_keys=("actions",),
        )
