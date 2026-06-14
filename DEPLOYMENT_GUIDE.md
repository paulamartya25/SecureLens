# DEPLOYMENT_GUIDE.md

## SecureLens Deployment Guide

Complete instructions for deploying SecureLens in different environments.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment (AWS/Azure/GCP)](#cloud-deployment)
4. [Production Checklist](#production-checklist)
5. [Security Configuration](#security-configuration)
6. [Monitoring & Logging](#monitoring--logging)
7. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.10+
- Virtual environment (venv)
- Git

### Setup Steps

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/securelens.git
   cd securelens
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run application**:
   ```bash
   python app.py
   ```

6. **Access web interface**:
   - Open browser at `http://localhost:5000`
   - Upload test X-ray image from `data/chest_xray/test/`

### Development with Hot Reload

```bash
export FLASK_DEBUG=True
export FLASK_ENV=development
python app.py
```

---

## Docker Deployment

### Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)

### Quick Start

```bash
# Build and run
docker-compose up --build

# Application runs on http://localhost:5000
```

### Build Custom Image

```bash
# Build image
docker build -t securelens:latest .

# Run container
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -v $(pwd)/client/uploads:/app/client/uploads \
  securelens:latest

# Run with GPU (if available)
docker run --gpus all -p 5000:5000 securelens:latest
```

### Docker Compose Customization

Edit `docker-compose.yml` to customize:
- Port mapping
- Environment variables
- Volume mounts
- Resource limits

```yaml
services:
  securelens:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## Cloud Deployment

### AWS Deployment (ECS/Fargate)

1. **Push to ECR**:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker tag securelens:latest <account>.dkr.ecr.<region>.amazonaws.com/securelens:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/securelens:latest
   ```

2. **Create ECS task definition** (from Docker image)

3. **Deploy service** on Fargate

### Azure Deployment (ACI/Container Instances)

```bash
az acr build --registry <registry-name> --image securelens:latest .
az container create \
  --resource-group <group-name> \
  --name securelens \
  --image <registry>.azurecr.io/securelens:latest \
  --cpu 2 --memory 4 \
  --ports 5000 \
  --environment-variables FLASK_ENV=production
```

### GCP Deployment (Cloud Run)

```bash
gcloud run deploy securelens \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars FLASK_ENV=production
```

---

## Production Checklist

Before deploying to production:

- [ ] **Security**
  - [ ] `debug=False` in production
  - [ ] `SECRET_KEY` is strong and unique
  - [ ] HTTPS/TLS enabled
  - [ ] CORS configured for specific origins
  - [ ] API key authentication enabled
  - [ ] Rate limiting enabled
  - [ ] Input validation on all endpoints

- [ ] **Performance**
  - [ ] Model cached in memory
  - [ ] CKKS context pre-initialized
  - [ ] Database connection pooling (if used)
  - [ ] CDN configured for static assets
  - [ ] Caching headers set correctly

- [ ] **Monitoring**
  - [ ] Logging configured (JSON format)
  - [ ] Error tracking enabled (Sentry, etc.)
  - [ ] Metrics exposed (Prometheus)
  - [ ] Health check endpoint working
  - [ ] Alerts configured for errors

- [ ] **Data**
  - [ ] Upload directory permissions correct
  - [ ] Uploads encrypted at rest (optional)
  - [ ] Audit logs enabled
  - [ ] Data retention policy enforced
  - [ ] Backup strategy in place

- [ ] **Documentation**
  - [ ] API documentation complete
  - [ ] Runbooks written
  - [ ] Incident response plan ready
  - [ ] Deployment procedure documented
  - [ ] Rollback procedure tested

---

## Security Configuration

### SSL/TLS

Generate self-signed certificate (testing only):

```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

Configure in app:

```python
app.run(
    ssl_context=('certs/cert.pem', 'certs/key.pem'),
    host='0.0.0.0',
    port=443
)
```

### Environment Variables (Production)

Set in `.env`:

```
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=<generate-strong-secret>
API_KEY_REQUIRED=True
CORS_ORIGINS=https://yourdomain.com
RATE_LIMIT_ENABLED=True
```

### Secrets Management

Use cloud secrets service:

```bash
# AWS Secrets Manager
aws secretsmanager create-secret --name securelens/api-key --secret-string "..."

# Azure Key Vault
az keyvault secret set --vault-name <name> --name api-key --value "..."

# GCP Secret Manager
gcloud secrets create securelens-api-key --data-file=- < api-key.txt
```

---

## Monitoring & Logging

### Logging

SecureLens logs to `logs/securelens.log`:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Prediction made", extra={"user": "...", "result": "..."})
```

### Structured Logging

Configure JSON logging:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_data)
```

### Metrics

Expose metrics for Prometheus:

```
/metrics endpoint returns:
- securelens_predictions_total (counter)
- securelens_prediction_latency_seconds (histogram)
- securelens_encryption_latency_seconds (histogram)
```

### Monitoring Tools

Recommended:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **ELK Stack**: Logging
- **Sentry**: Error tracking

---

## Troubleshooting

### Issue: CKKS Initialization Error

```
RuntimeError: CKKS context already initialized
```

**Solution**: Restart application, check for multiple workers

### Issue: Out of Memory

```
MemoryError: Cannot allocate memory for ciphertext
```

**Solution**:
- Reduce batch size
- Increase container memory limit
- Use GPU if available

### Issue: Slow Inference

Expected: 3-5 seconds for full inference

If slower:
- Check CPU usage (should be >90%)
- Verify model is loaded
- Check for disk I/O issues

### Issue: Upload Failures

```
413 Payload Too Large
```

**Solution**: Increase `UPLOAD_MAX_SIZE_MB` in `.env`

### Issue: CORS Errors

```
Access to XMLHttpRequest blocked by CORS policy
```

**Solution**: Check `CORS_ORIGINS` in configuration

---

## Performance Tuning

### Inference Optimization

```python
# Pre-load model and context
model.eval()
torch.no_grad()  # Disable gradients
# Batch multiple requests if possible
```

### Ciphertext Compression

CKKS ciphertexts can be compressed:

```python
import gzip
compressed = gzip.compress(ciphertext_bytes)
```

### Caching

Cache ResNet features for repeated uploads:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def extract_features(image_hash):
    # Expensive feature extraction
    pass
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily backup of models and logs
0 2 * * * tar -czf backups/securelens-$(date +\%Y\%m\%d).tar.gz \
  cloud_server/models/ logs/ > /dev/null 2>&1
```

### Restore from Backup

```bash
tar -xzf backups/securelens-20260610.tar.gz -C /app/
```

### Health Checks

Regular health monitoring:

```bash
curl http://localhost:5000/health
# Expected response: {"status": "ok"}
```

---

## Support & Issues

- **Documentation**: See [README.md](README.md)
- **Issues**: GitHub Issues
- **Security**: Email securelens-security@example.com
- **Community**: GitHub Discussions

---

**Last Updated**: June 2026  
**Version**: 1.0.0
