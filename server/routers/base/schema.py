import pydantic


class HealthResponse(pydantic.BaseModel):
    """HealthResponse Schema

    Attributes:
        project (str): project name.
        version (str): project version.
    """
    project: str
    version: str