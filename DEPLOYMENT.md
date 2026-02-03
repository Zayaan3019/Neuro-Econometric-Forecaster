# Production Deployment Guide 🚀

## Quick Reference

```bash
# Local Development
python serve_api.py                    # Start API server
python train_model.py                  # Train model
python evaluate_model.py               # Evaluate performance
python setup_check.py                  # System diagnostics

# Docker Deployment
docker-compose up -d                   # Start containerized service
docker-compose logs -f                 # View logs
docker-compose down                    # Stop service

# API Testing
curl http://localhost:8000/health      # Health check
curl http://localhost:8000/docs        # Interactive docs
```

---

## 📦 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t market-alpha-engine .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models_saved:/app/models_saved \
  -v $(pwd)/logs:/app/logs \
  --name alpha-engine \
  market-alpha-engine

# Check logs
docker logs -f alpha-engine

# Stop container
docker stop alpha-engine
```

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f market-alpha-engine

# Restart service
docker-compose restart

# Stop all services
docker-compose down
```

---

## ☁️ Cloud Deployment

### AWS EC2

```bash
# 1. Launch EC2 instance (t3.medium or larger)
# 2. SSH into instance
ssh -i your-key.pem ec2-user@your-ip

# 3. Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# 4. Clone repository
git clone your-repo-url
cd market-alpha-engine

# 5. Generate data and train model
python generate_sample_data.py
python train_model.py

# 6. Start service
docker-compose up -d

# 7. Configure security group (allow port 8000)
```

### Google Cloud Run

```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/market-alpha-engine

# 2. Deploy to Cloud Run
gcloud run deploy market-alpha-engine \
  --image gcr.io/PROJECT_ID/market-alpha-engine \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2

# 3. Get service URL
gcloud run services describe market-alpha-engine --region us-central1
```

### Azure Container Instances

```bash
# 1. Create resource group
az group create --name market-alpha-rg --location eastus

# 2. Build and push to ACR
az acr create --resource-group market-alpha-rg --name marketalphaacr --sku Basic
az acr build --registry marketalphaacr --image market-alpha-engine .

# 3. Deploy container
az container create \
  --resource-group market-alpha-rg \
  --name market-alpha-engine \
  --image marketalphaacr.azurecr.io/market-alpha-engine \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label market-alpha-api
```

---

## 🔐 Production Security

### API Authentication

Add JWT authentication to FastAPI endpoints:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict(request: PredictionRequest, token: dict = Depends(verify_token)):
    # Your prediction logic
    pass
```

### Environment Variables

Never commit sensitive data. Use `.env` file:

```bash
# .env
NEWS_API_KEY=your_api_key_here
JWT_SECRET=your_secret_here
DATABASE_URL=your_db_url_here
```

Load with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

### HTTPS/SSL

Use reverse proxy (nginx) with Let's Encrypt:

```nginx
# /etc/nginx/sites-available/alpha-engine
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📊 Monitoring & Logging

### Prometheus Metrics

Add to `serve_api.py`:

```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(request: PredictionRequest):
    prediction_counter.inc()
    with prediction_latency.time():
        # Your prediction logic
        pass

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "prediction_made",
    ticker=ticker,
    signal=signal,
    confidence=confidence,
    latency_ms=latency
)
```

### Log Aggregation

- **Local**: `tail -f logs/training.log`
- **CloudWatch**: AWS CloudWatch Logs
- **Stackdriver**: Google Cloud Logging
- **ELK Stack**: Elasticsearch + Logstash + Kibana

---

## 🚀 Performance Optimization

### GPU Acceleration

```python
# config.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Use GPU-enabled EC2 instances (p3.2xlarge) or Cloud GPU VMs.

### Model Optimization

```python
# Quantization (INT8)
import torch.quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX Export
torch.onnx.export(model, dummy_input, "model.onnx")

# TorchScript
model_scripted = torch.jit.script(model)
model_scripted.save("model_scripted.pt")
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_preprocessed_data(ticker: str, date: str):
    # Cache preprocessed data
    pass
```

### Async Workers

```bash
# Use Gunicorn with Uvicorn workers
gunicorn serve_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and push Docker
        run: |
          docker build -t market-alpha-engine .
          docker push your-registry/market-alpha-engine:latest
      
      - name: Deploy to server
        run: |
          ssh user@server 'docker pull your-registry/market-alpha-engine:latest && docker-compose up -d'
```

---

## 📈 Scaling Strategy

### Horizontal Scaling

```bash
# Docker Swarm
docker swarm init
docker stack deploy -c docker-compose.yml alpha-stack

# Kubernetes
kubectl create deployment alpha-engine --image=market-alpha-engine
kubectl scale deployment alpha-engine --replicas=3
kubectl expose deployment alpha-engine --port=8000 --type=LoadBalancer
```

### Load Balancing

```nginx
upstream alpha_backend {
    least_conn;
    server 10.0.0.1:8000 weight=3;
    server 10.0.0.2:8000 weight=2;
    server 10.0.0.3:8000 weight=1;
}

server {
    location / {
        proxy_pass http://alpha_backend;
    }
}
```

---

## 🛠️ Maintenance

### Model Retraining

Schedule periodic retraining:

```bash
# Crontab entry (retrain weekly)
0 2 * * 0 cd /app && python train_model.py

# Or use Airflow DAG for orchestration
```

### Backup Strategy

```bash
# Backup models and data daily
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf backup_$DATE.tar.gz models_saved/ data/
aws s3 cp backup_$DATE.tar.gz s3://your-bucket/backups/
```

### Health Monitoring

```bash
# Monitor endpoint with cron
*/5 * * * * curl -f http://localhost:8000/health || echo "API down" | mail -s "Alert" admin@example.com
```

---

## 📞 Support & Troubleshooting

### Common Issues

**1. Model not loading**
```bash
# Check model file exists
ls -lh models_saved/best_model.pt

# Verify permissions
chmod 644 models_saved/best_model.pt
```

**2. Out of memory**
```python
# Reduce batch size
Config.BATCH_SIZE = 16

# Clear cache
torch.cuda.empty_cache()
```

**3. API timeout**
```python
# Increase timeout in Uvicorn
uvicorn serve_api:app --timeout-keep-alive 60
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python serve_api.py
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Production Guide](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

---

## ✅ Pre-Deployment Checklist

- [ ] Model trained and validated
- [ ] API endpoints tested
- [ ] Environment variables configured
- [ ] Docker image built successfully
- [ ] Security measures implemented
- [ ] Monitoring setup configured
- [ ] Backup strategy in place
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team trained on operations

---

<div align="center">
  <sub>Ready for Production! 🎉</sub>
</div>
