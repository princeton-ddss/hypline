import typer


def split_csv(value: str | None, param_hint: str | None = None) -> list[str] | None:
    if value is None:
        return None
    if any(c.isspace() for c in value):
        raise typer.BadParameter(
            "must not contain whitespace (e.g., 01,02,03)", param_hint=param_hint
        )
    items = value.split(",")
    if any(not v for v in items):
        raise typer.BadParameter("empty value between commas", param_hint=param_hint)
    if len(items) != len(set(items)):
        raise typer.BadParameter("duplicate values not allowed", param_hint=param_hint)
    return items
