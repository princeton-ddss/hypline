from pydantic import BaseModel, PositiveFloat, PositiveInt

from .enums import CompCorMethod, CompCorTissue


class CompCorOptions(BaseModel):
    n_comps: PositiveInt | PositiveFloat = 5
    tissue: CompCorTissue | None = None


class ConfoundMetadata(BaseModel):
    Method: CompCorMethod
    Retained: bool | None = None
    Mask: CompCorTissue | None = None
    SingularValue: float | None = None
    VarianceExplained: float | None = None
    CumulativeVarianceExplained: float | None = None


class ModelSpec(BaseModel):
    confounds: list[str]
    custom_confounds: list[str] | None = None
    aCompCor: list[CompCorOptions] | None = None
    tCompCor: list[CompCorOptions] | None = None


class Config(BaseModel):
    model_specs: dict[str, ModelSpec]
