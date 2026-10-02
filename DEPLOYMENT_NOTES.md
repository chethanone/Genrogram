# GENGROGRAM deployment notes

## Frontend
Set `NEXT_PUBLIC_API_URL` to the deployed FastAPI URL.

## Backend
Set `FRONTEND_ORIGIN` to the deployed frontend origin. Multiple comma-separated origins are accepted.

## Upload limit
The web client limits audio uploads to 4 MB because Vercel Function request payloads are limited to 4.5 MB.

## Production model
Only `models/production/gengrogram_custom_cnn_20class_production.pt` is required for inference.
