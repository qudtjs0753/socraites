# Level 4: 프로덕션 K8S (4주)

## 목차
- [학습 목표](#학습-목표)
- [9주차: Kind 클러스터 + Docker RAG](#9주차-kind-클러스터--docker-rag)
- [10주차: Chroma DB K8S 배포](#10주차-chroma-db-k8s-배포)
- [11주차: Elasticsearch ECK 배포](#11주차-elasticsearch-eck-배포)
- [12주차: 모니터링 + 스케일링 + MLOps](#12주차-모니터링--스케일링--mlops)
- [핵심 실습 프로젝트](#핵심-실습-프로젝트)
- [체크리스트](#체크리스트)

---

## 학습 목표

4주 후 달성 목표:
- Kind 로컬 K8S 클러스터에서 완전한 RAG 서비스 운영
- Chroma DB와 Elasticsearch를 K8S에 안정적으로 배포
- Prometheus + Grafana로 RAG 성능 모니터링

---

## 9주차: Kind 클러스터 + Docker RAG

### Kind 클러스터 설계 (RAG 서비스용)

```
Kind Cluster: rag-study
├── control-plane (포트포워딩: 8080→80)
├── worker-1 (RAG API + Chroma)
└── worker-2 (Elasticsearch)
```

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: rag-study
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080  # RAG API NodePort
        hostPort: 8080
      - containerPort: 30090  # Grafana NodePort
        hostPort: 3000
  - role: worker
    labels:
      workload: rag-app     # RAG API + Chroma 배치용
    extraMounts:
      - hostPath: /tmp/rag-chroma-data
        containerPath: /chroma-data
  - role: worker
    labels:
      workload: search       # Elasticsearch 배치용
    extraMounts:
      - hostPath: /tmp/rag-es-data
        containerPath: /es-data
```

```bash
# 클러스터 생성
mkdir -p /tmp/rag-chroma-data /tmp/rag-es-data
kind create cluster --config kind-config.yaml

# 네임스페이스 생성
kubectl create namespace rag-system
kubectl create namespace search
kubectl create namespace monitoring
```

### RAG FastAPI 서비스 구현

```python
# main.py — 프로덕션 RAG API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time, os

app = FastAPI(title="RAG API")

# Prometheus 메트릭
REQUEST_COUNT = Counter("rag_requests_total", "총 요청 수", ["status"])
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds", "요청 처리 시간",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.time()
    try:
        result = rag_chain.invoke({"query": request.question})
        latency = (time.time() - start) * 1000
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start)
        return QueryResponse(
            answer=result["result"],
            sources=[d.metadata.get("source", "") for d in result["source_documents"]],
            latency_ms=latency
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
```

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```bash
# Kind에 이미지 로드
docker build -t rag-api:v1 .
kind load docker-image rag-api:v1 --name rag-study
```

### K8S ConfigMap + Secret 생성 (사내 LLM 설정 관리)

```bash
# URL·모델명은 ConfigMap (비밀 아님)
kubectl create configmap rag-config \
  --from-literal=llm-base-url="${LLM_BASE_URL}" \
  --from-literal=llm-model="${LLM_MODEL}" \
  --from-literal=embed-base-url="${EMBED_BASE_URL}" \
  --from-literal=embed-model="${EMBED_MODEL}" \
  -n rag-system

# API 키는 Secret
kubectl create secret generic rag-secrets \
  --from-literal=llm-api-key="${LLM_API_KEY}" \
  --from-literal=embed-api-key="${EMBED_API_KEY}" \
  -n rag-system
```

---

## 10주차: Chroma DB K8S 배포

### Chroma StatefulSet + 영속성

```yaml
# chroma-deployment.yaml
---
apiVersion: v1
kind: Service
metadata:
  name: chroma-svc
  namespace: rag-system
spec:
  selector:
    app: chroma
  ports:
    - port: 8000
      targetPort: 8000
  clusterIP: None  # Headless Service (StatefulSet용)
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chroma
  namespace: rag-system
spec:
  serviceName: chroma-svc
  replicas: 1
  selector:
    matchLabels:
      app: chroma
  template:
    metadata:
      labels:
        app: chroma
    spec:
      nodeSelector:
        workload: rag-app   # worker-1에 배치
      containers:
        - name: chroma
          image: chromadb/chroma:0.5.0
          ports:
            - containerPort: 8000
          env:
            - name: CHROMA_SERVER_HOST
              value: "0.0.0.0"
            - name: CHROMA_SERVER_CORS_ALLOW_ORIGINS
              value: '["*"]'
          volumeMounts:
            - name: chroma-data
              mountPath: /chroma/chroma
          resources:
            requests:
              memory: "512Mi"
              cpu: "200m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          readinessProbe:
            httpGet:
              path: /api/v1/heartbeat
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
  volumeClaimTemplates:
    - metadata:
        name: chroma-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: standard   # Kind 기본 StorageClass
        resources:
          requests:
            storage: 10Gi
---
# RAG API가 Chroma에 연결하는 Service
apiVersion: v1
kind: Service
metadata:
  name: chroma-access
  namespace: rag-system
spec:
  selector:
    app: chroma
  ports:
    - port: 8000
      targetPort: 8000
```

```bash
kubectl apply -f chroma-deployment.yaml
kubectl wait --for=condition=ready pod/chroma-0 -n rag-system --timeout=120s

# 데이터 적재 확인
kubectl port-forward svc/chroma-access 8001:8000 -n rag-system &
python scripts/load_data.py  # 로컬에서 Chroma에 문서 인덱싱
```

---

## 11주차: Elasticsearch ECK 배포

### ECK (Elastic Cloud on Kubernetes) 설치

```bash
# CRD 설치
kubectl create -f https://download.elastic.co/downloads/eck/2.12.1/crds.yaml

# ECK 오퍼레이터 설치
kubectl apply -f https://download.elastic.co/downloads/eck/2.12.1/operator.yaml

# 오퍼레이터 준비 확인
kubectl wait --for=condition=ready pod -l control-plane=elastic-operator \
  -n elastic-system --timeout=120s
```

### Elasticsearch 클러스터 배포 (Kind 최적화)

```yaml
# elasticsearch.yaml
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: rag-es
  namespace: search
spec:
  version: 8.13.0
  nodeSets:
    - name: default
      count: 1  # Kind 환경 단일 노드
      config:
        node.store.allow_mmap: false  # Kind 필수 (mmap 제한)
        xpack.security.enabled: false  # 개발 환경 보안 비활성화
      podTemplate:
        spec:
          nodeSelector:
            workload: search   # worker-2에 배치
          initContainers:
            - name: sysctl
              securityContext:
                privileged: true
              command: ["sh", "-c", "sysctl -w vm.max_map_count=262144"]
          containers:
            - name: elasticsearch
              resources:
                requests:
                  memory: 1Gi
                  cpu: 500m
                limits:
                  memory: 2Gi
                  cpu: 2000m
              env:
                - name: ES_JAVA_OPTS
                  value: "-Xms512m -Xmx512m"
```

```bash
kubectl apply -f elasticsearch.yaml

# 준비 확인
kubectl get elasticsearch -n search
# NAME     HEALTH   NODES   VERSION   PHASE
# rag-es   green    1       8.13.0    Ready

# 접근 확인
kubectl port-forward svc/rag-es-es-http 9200:9200 -n search &
curl -X GET "localhost:9200/_cat/indices"
```

### Kibana (선택) + Nori 플러그인

```bash
# Nori 플러그인은 ECK에서 initContainer로 설치
# elasticsearch.yaml에 추가:
spec:
  nodeSets:
    - name: default
      podTemplate:
        spec:
          initContainers:
            - name: install-nori
              command:
                - sh
                - -c
                - |
                  bin/elasticsearch-plugin install analysis-nori --batch
              volumeMounts:
                - name: elastic-internal-transport-certificates
                  mountPath: /usr/share/elasticsearch/config/certs
```

---

## 12주차: 모니터링 + 스케일링 + MLOps

### kube-prometheus-stack 설치

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm repo update

# Kind 환경에 맞게 NodePort 사용
cat > prometheus-values.yaml << 'EOF'
grafana:
  service:
    type: NodePort
    nodePort: 30090
  adminPassword: "rag-study-2024"
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: 'default'
          folder: 'RAG'
          type: file
          options:
            path: /var/lib/grafana/dashboards/default
prometheus:
  prometheusSpec:
    podMonitorSelectorNilUsesHelmValues: false
    serviceMonitorSelectorNilUsesHelmValues: false
EOF

helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values prometheus-values.yaml
```

### RAG API ServiceMonitor (Prometheus 자동 수집)

```yaml
# rag-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: rag-api-monitor
  namespace: rag-system
  labels:
    release: monitoring  # Prometheus가 발견하는 레이블
spec:
  selector:
    matchLabels:
      app: rag-api
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

### Grafana 대시보드 (RAG 핵심 지표)

Grafana에서 Import할 대시보드 패널:

```json
{
  "panels": [
    {
      "title": "RAG 요청 처리량 (req/s)",
      "targets": [{ "expr": "rate(rag_requests_total[1m])" }]
    },
    {
      "title": "RAG P95 레이턴시",
      "targets": [{ "expr": "histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))" }]
    },
    {
      "title": "RAG 에러율",
      "targets": [{ "expr": "rate(rag_requests_total{status='error'}[1m]) / rate(rag_requests_total[1m])" }]
    }
  ]
}
```

### HPA (Horizontal Pod Autoscaler)

```yaml
# rag-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 70
```

### MLflow 실험 추적 (선택)

```bash
# MLflow 간단 배포
helm install mlflow community-charts/mlflow \
  --namespace rag-system \
  --set service.type=NodePort
```

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_name="rag-experiment-v1"):
    mlflow.log_params({
        "chunk_size": 500,
        "overlap": 50,
        "retriever_type": "hybrid",
        "llm": os.getenv("LLM_MODEL", "unknown"),
    })
    mlflow.log_metrics({
        "faithfulness": 0.88,
        "answer_relevancy": 0.82,
        "context_recall": 0.86,
        "p95_latency_ms": 1200
    })
```

---

## 핵심 실습 프로젝트

**"Kind 위에서 완전한 RAG 서비스 운영"**

구성:
- Kind 3노드 클러스터
- RAG API (FastAPI) — 2 replica, HPA 설정
- Chroma DB — StatefulSet, 10Gi PVC
- Elasticsearch — ECK, Nori 플러그인
- Prometheus + Grafana 모니터링

성공 기준:
- [ ] `curl localhost:8080/query -d '{"question":"RAG란?"}'` 응답 성공
- [ ] Grafana 대시보드에서 RAG 레이턴시 실시간 확인
- [ ] 부하 테스트 시 HPA로 자동 스케일아웃 확인

---

## 체크리스트

**인프라 구성**
- [ ] Kind 3노드 클러스터 생성 + 네임스페이스 분리
- [ ] Chroma StatefulSet + PVC 영속 배포
- [ ] Elasticsearch ECK + Nori 플러그인 배포

**서비스 배포**
- [ ] RAG FastAPI 도커 이미지 빌드 + Kind 로드
- [ ] K8S Deployment + Service + HPA 배포
- [ ] K8S Secret(API 키) + ConfigMap(URL·모델명) 분리 관리

**모니터링**
- [ ] Prometheus RAG 메트릭 수집 확인
- [ ] Grafana 대시보드 (레이턴시, 처리량, 에러율) 완성
- [ ] 부하 테스트 시 HPA 스케일링 동작 확인

**운영 능력**
- [ ] 포드 재시작 시 Chroma 데이터 보존 확인
- [ ] kubectl rollout를 통한 무중단 배포
- [ ] 장애 시나리오 (포드 강제 종료) 복구 확인
