---
name: fe-deploy
description: "프론트엔드 프로젝트를 Vercel에 배포하고 GitHub Actions CI/CD 파이프라인을 구성하는 스킬. vercel.json, .github/workflows, 환경 변수 설정, 도메인 연결을 처리한다. '배포해줘', 'Vercel에 올려줘', 'CI/CD 구성해줘', 'GitHub Actions 설정해줘', '환경 변수 어떻게 관리해', '자동 배포 설정해줘' 요청 시 반드시 이 스킬을 사용할 것."
---

# 프론트엔드 배포 스킬

## 목적

완성된 프로젝트를 실제 사용자가 접근할 수 있는 URL로 배포한다. 로컬 빌드 → 수동 배포 → 자동 배포 순서로 진행한다.

## 배포 전 체크리스트

```
□ npm run build   → 로컬 빌드 성공 확인
□ npm run test    → 테스트 전체 통과 확인
□ .gitignore      → .env* 파일 포함 확인
□ .env.example   → 필요한 환경 변수 목록 작성
□ README.md      → 실행 방법 기록
```

## Step 1: 로컬 빌드 확인

```bash
npm run build
# 오류 없이 완료되면 다음 단계 진행

npm run start
# 프로덕션 모드로 로컬 실행 확인 (http://localhost:3000)
```

**흔한 빌드 오류:**
- TypeScript 타입 오류: `tsc --noEmit`으로 미리 확인
- 환경 변수 누락: `.env.local` 값이 Vercel에 등록됐는지 확인
- Server/Client 컴포넌트 혼용: `'use client'` 없는 컴포넌트에서 브라우저 API 사용

## Step 2: Vercel 수동 배포

```bash
# Vercel CLI 설치 (최초 1회)
npm install -g vercel

# 로그인
vercel login

# 배포 (프로젝트 루트에서)
vercel

# 프로덕션 배포
vercel --prod
```

### vercel.json (선택)

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs"
}
```

## Step 3: GitHub 연동 자동 배포

Vercel 대시보드 → Import Project → GitHub 저장소 선택

**자동 배포 동작:**
- `main` 브랜치 push → 프로덕션 자동 배포
- 다른 브랜치 push → Preview URL 자동 생성 (PR 리뷰에 유용)

### 환경 변수 설정

Vercel 대시보드 → Settings → Environment Variables:
```
NEXT_PUBLIC_API_URL    → Production, Preview, Development 모두 설정
DATABASE_URL           → Production만 설정
```

`NEXT_PUBLIC_` 접두사: 브라우저에서도 접근 가능 (민감한 정보 사용 금지)

## Step 4: GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Type check
        run: npx tsc --noEmit

      - name: Lint
        run: npm run lint

      - name: Test
        run: npm run test -- --run

      - name: Build
        run: npm run build
```

## 환경 변수 파일 관리

```bash
# .env.local (로컬 전용 — Git에 포함하지 않음)
NEXT_PUBLIC_API_URL=http://localhost:3001
DATABASE_URL=postgresql://localhost:5432/mydb

# .env.example (팀 공유용 템플릿 — Git에 포함)
NEXT_PUBLIC_API_URL=        # 공개 API 엔드포인트
DATABASE_URL=               # PostgreSQL 연결 문자열
```

```gitignore
# .gitignore에 반드시 포함
.env
.env.local
.env.production
```

## 배포 후 확인

```
□ 배포 URL 접속 → 정상 렌더링
□ API 연동 → 데이터 정상 로드
□ Vercel 대시보드 → 빌드 로그 오류 없음
□ Vercel Analytics → Core Web Vitals 초록불 (선택)
```

## 롤백 방법

Vercel 대시보드 → Deployments → 이전 배포 선택 → "Promote to Production"

## 흔한 배포 오류

| 오류 | 원인 | 해결 |
|------|------|------|
| 빌드 실패 | 타입 오류, 환경 변수 누락 | `npm run build` 로컬 확인 |
| 500 Internal Server Error | 서버 컴포넌트에서 DB 연결 실패 | Vercel 환경 변수 확인 |
| 빈 화면 | `'use client'` 누락 | 브라우저 API 사용 컴포넌트에 추가 |
| CORS 오류 | 외부 API 도메인 설정 | `next.config.js`의 `rewrites` 활용 |
