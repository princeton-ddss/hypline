from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from .enums import CompCorMask, CompCorMethod


class CompCorOptions(BaseModel):
    n_comps: PositiveInt | PositiveFloat = 5
    mask: CompCorMask | None = None


class ConfoundMetadata(BaseModel):
    Method: CompCorMethod
    Retained: bool | None = None
    Mask: CompCorMask | None = None
    SingularValue: float | None = None
    VarianceExplained: float | None = None
    CumulativeVarianceExplained: float | None = None


class ModelSpec(BaseModel):
    confounds: list[str] = Field(min_length=1)
    custom_confounds: list[str] | None = None
    aCompCor: list[CompCorOptions] | None = None
    tCompCor: list[CompCorOptions] | None = None


class Config(BaseModel):
    model_specs: dict[str, ModelSpec]
