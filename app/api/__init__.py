"""
API module.

The FastAPI app lives at the repository root (serve_api.py), which is the
entrypoint actually used by Dockerfile (`CMD ["python", "serve_api.py"]`).
This package previously also contained a second, unused, duplicate FastAPI
app (routes.py) with a hardcoded `input_dim=50` placeholder that was never
filled in and referenced the old, incompatible checkpoint format -- it has
been removed rather than fixed in place, since maintaining two divergent
copies of the same API is itself a maintainability hazard.
"""
