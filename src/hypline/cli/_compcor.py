import typer

from ._utils import split_csv


def parse_compcor(value: str | None):
    """Parse `--compcor` tokens into CompCorOptions.

    Grammar: comma-separated `[a|t]:[mask]:[n]` tokens, each with fixed 3-field
    arity. `a`=aCompCor (mask required), `t`=tCompCor (mask slot must be empty
    — tCompCor is not mask-restricted). `n` is a positive int (top-N) or float
    (variance fraction). Validation is strict and fails fast at this boundary.
    """
    from hypline.confounds.fmriprep import CompCorMask, CompCorMethod, CompCorOptions

    hint = "--compcor"
    if value is None:
        return []
    tokens = split_csv(value, param_hint=hint)
    assert tokens is not None  # split_csv returns None only for None input

    methods = {"a": CompCorMethod.ANATOMICAL, "t": CompCorMethod.TEMPORAL}
    options: list[CompCorOptions] = []
    for token in tokens:
        fields = token.split(":")
        if len(fields) != 3:
            raise typer.BadParameter(
                f"'{token}' must have 3 colon-separated fields ([a|t]:[mask]:[n])",
                param_hint=hint,
            )
        method_str, mask_str, n_str = fields

        if method_str not in methods:
            raise typer.BadParameter(
                f"'{token}' method must be 'a' or 't', got '{method_str}'",
                param_hint=hint,
            )
        method = methods[method_str]

        mask: CompCorMask | None = None
        if method is CompCorMethod.ANATOMICAL:
            if not mask_str:
                raise typer.BadParameter(
                    f"'{token}' aCompCor requires a mask (e.g., a:CSF:5)",
                    param_hint=hint,
                )
            try:
                mask = CompCorMask(mask_str)
            except ValueError:
                allowed = ", ".join(m.value for m in CompCorMask)
                raise typer.BadParameter(
                    f"'{token}' mask must be one of: {allowed}", param_hint=hint
                ) from None
        elif mask_str:
            raise typer.BadParameter(
                f"'{token}' tCompCor is not mask-restricted — leave mask empty "
                "(e.g., t::10)",
                param_hint=hint,
            )

        n_comps = parse_n_comps(n_str, token=token, hint=hint)
        options.append(CompCorOptions(method=method, n_comps=n_comps, mask=mask))

    return options


def parse_n_comps(n_str: str, *, token: str, hint: str) -> int | float:
    """Parse the `n` field: a positive int (top-N) or positive float (variance)."""
    if not n_str:
        raise typer.BadParameter(
            f"'{token}' missing n (e.g., a:CSF:5)", param_hint=hint
        )
    try:
        n_comps: int | float = int(n_str)
    except ValueError:
        try:
            n_comps = float(n_str)
        except ValueError:
            raise typer.BadParameter(
                f"'{token}' n must be a number, got '{n_str}'", param_hint=hint
            ) from None
    if n_comps <= 0:
        raise typer.BadParameter(f"'{token}' n must be positive", param_hint=hint)
    return n_comps
