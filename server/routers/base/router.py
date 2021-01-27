import fastapi
import bert_repro
from server.routers.base.schema import HealthResponse

PROJECT_NAME = "bert-repro"

router = fastapi.APIRouter()

@router.get('/health', response_model=HealthResponse)
async def health():
    """Health Check endpoint
    """
    return HealthResponse(
        project=PROJECT_NAME, 
        version=bert_repro.__version__
    )
    
