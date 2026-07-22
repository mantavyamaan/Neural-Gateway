import asyncio
from app.api.schemas import RouteRequest, RequestConstraintsIn
from app.core.router import route
import app.core.router as router_mod

def mock_generate(*args, **kwargs):
    model = kwargs.get('model') or args[0]
    print(f'Model: {model["name"]}')
    return original_generate(*args, **kwargs)

original_generate = router_mod.generate_single_model_plan
router_mod.generate_single_model_plan = mock_generate

async def main():
    payload = RouteRequest(
        prompt='generate me a code for my highly advanced portfolio website',
        estimated_tokens=14,
        estimated_output_tokens=50,
        request_constraints=RequestConstraintsIn(max_cost_usd=0.0029, max_latency_ms=30000.0)
    )
    res = route(
        prompt=payload.prompt,
        estimated_tokens=payload.estimated_tokens,
        estimated_output_tokens=payload.estimated_output_tokens,
        request_constraints=payload.request_constraints,
        profile_name='balanced'
    )
    print(res.decision_record.get('plan_constraints'))

asyncio.run(main())
