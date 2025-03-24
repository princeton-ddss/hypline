from pydantic import BaseModel


class ModelSpec(BaseModel):
    confounds: list[str]
    custom_confounds: list[str] | None = None
    aCompCor: list[dict] | None = None
    tCompCor: list[dict] | None = None


class Config(BaseModel):
    model_specs: dict[str, ModelSpec]
