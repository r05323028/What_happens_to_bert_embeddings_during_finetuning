import fastapi

import bert_repro
import server.routers.base.router as base_router


def create_app() -> fastapi.FastAPI:
    '''Creates fastapi app
    '''
    app = fastapi.FastAPI(
        title="bert_repro",
        version=bert_repro.__version__
    )

    app.include_router(base_router.router)

    return app