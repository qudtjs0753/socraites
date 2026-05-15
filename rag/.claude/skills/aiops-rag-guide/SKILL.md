---
name: aiops-rag-guide
description: "K8S(Kind) 기반 RAG 인프라 배포 가이드 스킬. Kind 클러스터 설정, RAG 서비스 K8S 배포, Chroma StatefulSet, Elasticsearch ECK, Helm 차트, Prometheus/Grafana 모니터링, HPA 스케일링, RAG 도커파일 작성 요청 시 이 스킬을 사용하라. '쿠버네티스에 RAG 배포', 'Kind 클러스터', 'RAG 컨테이너화', 'ECK 설치', 'RAG 모니터링' 등 모든 K8S/인프라 관련 RAG 요청에 이 스킬을 사용."
---

# AIOps RAG 인프라 가이드 스킬

K8S(Kind) 환경에서 RAG 서비스를 배포하고 운영하는 AIOps 가이드.

## Kind 클러스터 설정

### 멀티노드 Kind 클러스터 (RAG 용도)

```yaml
# kind-rag-cluster.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: rag-study
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 8080   # RAG API 접근용
        protocol: TCP
  - role: worker            # Chroma DB 노드
    extraMounts:
      - hostPath: /tmp/chroma-data
        containerPath: /data
  - role: worker            # Elasticsearch 노드
    extraMounts:
      - hostPath: /tmp/es-data
        containerPath: /data
```

```bash
kind create cluster --config kind-rag-cluster.yaml
kubectl cluster-info --context kind-rag-study
```

## RAG API 컨테이너화

### Dockerfile (FastAPI + LangChain)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 의존성 분리 설치 (캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 환경변수는 K8S Secret/ConfigMap으로 주입
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### K8S Deployment

```yaml
# rag-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
      annotations:
        prometheus.io/scrape: "true"  # Prometheus 자동 수집
        prometheus.io/port: "8000"
    spec:
      containers:
        - name: rag-api
          image: rag-api:latest
          imagePullPolicy: Never       # Kind 로컬 이미지
          ports:
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: rag-secrets
                  key: openai-api-key
            - name: CHROMA_HOST
              value: "chroma-svc.rag-system.svc.cluster.local"
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
```

## Chroma DB K8S 배포

```yaml
# chroma-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chroma
  namespace: rag-system
spec:
  serviceName: chroma
  replicas: 1
  selector:
    matchLabels:
      app: chroma
  template:
    metadata:
      labels:
        app: chroma
    spec:
      containers:
        - name: chroma
          image: chromadb/chroma:latest
          ports:
            - containerPort: 8000
          env:
            - name: CHROMA_SERVER_HOST
              value: "0.0.0.0"
          volumeMounts:
            - name: chroma-storage
              mountPath: /chroma/chroma
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "1Gi"
              cpu: "500m"
  volumeClaimTemplates:
    - metadata:
        name: chroma-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 5Gi
```

## Elasticsearch ECK 배포

```bash
# ECK 오퍼레이터 설치
kubectl create -f https://download.elastic.co/downloads/eck/2.12.1/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.12.1/operator.yaml
```

```yaml
# elasticsearch-cluster.yaml
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: rag-es
  namespace: search
spec:
  version: 8.13.0
  nodeSets:
    - name: default
      count: 1            # Kind 환경 최소 구성
      config:
        node.store.allow_mmap: false   # Kind 환경 필수 설정
      podTemplate:
        spec:
          containers:
            - name: elasticsearch
              resources:
                requests:
                  memory: 1Gi
                  cpu: 500m
                limits:
                  memory: 2Gi
```

## 모니터링 스택

```bash
# Prometheus + Grafana (kube-prometheus-stack)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.service.type=NodePort \
  --set grafana.service.nodePort=32000
```

### RAG 커스텀 메트릭 (FastAPI)

```python
from prometheus_client import Histogram, Counter

# RAG 쿼리 레이턴시 측정
RAG_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "RAG 쿼리 처리 시간",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

RETRIEVAL_COUNT = Counter(
    "rag_retrieval_total",
    "RAG 검색 횟수",
    ["retriever_type"]  # chroma, elasticsearch, hybrid
)
```

## HPA 자동 스케일링

```yaml
# rag-api-hpa.yaml
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
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

## Kind 로컬 이미지 로딩

```bash
# 빌드 후 Kind 클러스터에 로드 (레지스트리 없이)
docker build -t rag-api:latest .
kind load docker-image rag-api:latest --name rag-study
```

## 트러블슈팅 체크리스트

- [ ] `kubectl get nodes` — 모든 노드 Ready 상태 확인
- [ ] `kubectl top nodes` — 리소스 사용량 확인 (Kind 메모리 부족 주의)
- [ ] Chroma 접근: `kubectl port-forward svc/chroma-svc 8001:8000`
- [ ] ES 접근: `kubectl port-forward svc/rag-es-es-http 9200:9200`
- [ ] RAG API 로그: `kubectl logs -l app=rag-api -f`
