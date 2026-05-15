---
name: fe-workshop
description: "프론트엔드 프로젝트 생성, 아키텍처 구성, 학습 예제 코드 작성, 개발 도구 설정을 수행하는 스킬. React/Next.js 프로젝트 셋업, TanStack Query와 Tailwind 통합, ESLint/Prettier/TypeScript 설정을 실제 동작하는 코드로 구현한다. '프로젝트 만들어줘', '예제 코드 보여줘', '투두앱 만들어줘', 'Next.js 셋업해줘', 'ESLint 설정해줘', 'TanStack Query 연결해줘', 'Tailwind 설정해줘', '폴더 구조 잡아줘' 요청 시 반드시 이 스킬을 사용할 것."
---

# 프론트엔드 워크샵 스킬

## 목적

학습자가 직접 실행해보면서 배울 수 있는 프로젝트와 예제 코드를 작성한다. 모든 코드는 `npm run dev`로 바로 실행 가능해야 한다.

## 프로젝트 초기화

### Next.js 프로젝트 생성

```bash
npx create-next-app@latest my-app \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --src-dir \
  --import-alias "@/*"
```

### 핵심 의존성 추가

```bash
# TanStack Query
npm install @tanstack/react-query @tanstack/react-query-devtools

# 유틸리티 (선택)
npm install clsx tailwind-merge
```

## 표준 폴더 구조

```
src/
├── app/
│   ├── layout.tsx        # HTML 기본 틀 + Provider 감싸기
│   ├── page.tsx          # 홈 페이지 (/)
│   ├── providers.tsx     # QueryClientProvider (Client Component)
│   └── [feature]/
│       └── page.tsx
├── components/
│   ├── ui/               # Button, Input 등 재사용 UI
│   └── [feature]/        # 특정 기능 전용 컴포넌트
├── hooks/                # 재사용 Custom Hooks
├── lib/
│   ├── api.ts            # API 호출 함수
│   └── utils.ts          # cn() 같은 헬퍼
└── types/
    └── index.ts          # 공통 타입 정의
```

## TanStack Query 기본 설정

```tsx
// src/app/providers.tsx
'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { useState } from 'react';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
```

```tsx
// src/app/layout.tsx
import { Providers } from './providers';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

## 학습 예제 설계 패턴

각 주제의 예제는 5단계 복잡도로 설계한다:

```
Step 1: 정적 UI       → Tailwind로 레이아웃만
Step 2: 상태 추가     → useState로 인터랙션
Step 3: 서버 데이터   → useQuery로 API 연동
Step 4: 데이터 변경   → useMutation으로 CRUD
Step 5: 페이지 분리   → Next.js 라우팅
```

### 예제 1: Counter (useState 이해용)

```tsx
// src/app/counter/page.tsx
'use client';

import { useState } from 'react';

export default function CounterPage() {
  const [count, setCount] = useState(0);

  return (
    <div className="flex flex-col items-center gap-4 p-8">
      <h1 className="text-2xl font-bold">카운터: {count}</h1>
      <div className="flex gap-2">
        <button
          onClick={() => setCount(count - 1)}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          -1
        </button>
        <button
          onClick={() => setCount(count + 1)}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          +1
        </button>
      </div>
    </div>
  );
}
```

**학습 포인트**: state 변경 → 화면 재렌더링 관계

### 예제 2: Todo 앱 (useQuery + useMutation)

공개 API(`jsonplaceholder.typicode.com`)를 활용한 실습. 상세 구현은 `references/react-nextjs.md`의 "Todo 예제" 섹션 참조.

## 자주 쓰는 Tailwind 패턴

```tsx
// 중앙 정렬 컨테이너
<div className="max-w-2xl mx-auto px-4 py-8">

// 카드
<div className="rounded-lg border bg-white p-4 shadow-sm">

// 버튼 (primary)
<button className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">

// 입력 필드
<input className="w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">

// 반응형 그리드
<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
```

## ESLint + Prettier 설정

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "all",
  "printWidth": 100
}
```

```js
// eslint.config.js (Next.js 기본값에 추가)
import { FlatCompat } from '@eslint/eslintrc';

const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const eslintConfig = [
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  {
    rules: {
      'react-hooks/exhaustive-deps': 'warn',
      'no-console': ['warn', { allow: ['error'] }],
    },
  },
];

export default eslintConfig;
```

## 체크포인트 형식

각 예제 완성 후 학습자에게 안내할 체크포인트:

```
✅ 성공 기준:
- [ ] `npm run dev` 후 http://localhost:3000 접속 가능
- [ ] [기능 A]가 동작함
- [ ] [기능 B]가 동작함

🔧 다음 확장 과제:
1. [스스로 해볼 수 있는 기능 추가 아이디어]
2. [조금 더 도전적인 기능]
```

## 기술 스택별 상세 가이드

- React + Next.js 패턴: `references/react-nextjs.md`
- TanStack Query + Tailwind: `references/tanstack-tailwind.md`
- 개발 도구 설정: `references/tooling.md`
