---
name: fe-quality
description: "프론트엔드 테스트 코드 작성과 코드 품질 도구 설정을 수행하는 스킬. Vitest와 Testing Library를 사용한 컴포넌트/훅 테스트, MSW API 모킹, ESLint/Prettier/Husky 설정을 담당한다. '테스트 코드 써줘', '테스트 추가해줘', '린터 설정해줘', 'Prettier 설정해줘', 'Husky 설정해줘', '테스트 환경 구성해줘', '커버리지 설정' 요청 시 반드시 이 스킬을 사용할 것."
---

# 프론트엔드 품질 스킬

## 목적

테스트를 처음 접하는 학습자가 "무엇을, 왜 테스트하는가"를 이해하고, 실제 테스트를 작성할 수 있도록 안내한다.

## 테스트 환경 설정

### Vitest + Testing Library 설치

```bash
npm install -D vitest @vitest/ui jsdom
npm install -D @testing-library/react @testing-library/user-event @testing-library/jest-dom
npm install -D msw
```

### vitest.config.ts

```ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### src/test/setup.ts

```ts
import '@testing-library/jest-dom';
```

### package.json scripts 추가

```json
{
  "scripts": {
    "test": "vitest",
    "test:ui": "vitest --ui",
    "coverage": "vitest run --coverage"
  }
}
```

## 테스트 작성 원칙

### 테스트 대상 우선순위

```
1순위: 사용자 인터랙션 (클릭, 입력, 제출)
2순위: 조건부 렌더링 (로그인/비로그인, 로딩/에러/성공)
3순위: Custom Hooks 로직
4순위: 유틸리티 함수
```

### 기본 테스트 구조 (AAA 패턴)

```tsx
it('버튼 클릭 시 카운트가 1 증가한다', async () => {
  // Arrange: 준비
  render(<Counter initialCount={0} />);

  // Act: 실행
  await userEvent.click(screen.getByRole('button', { name: '+1' }));

  // Assert: 확인
  expect(screen.getByText('카운터: 1')).toBeInTheDocument();
});
```

### 컴포넌트 테스트 예시

```tsx
// Counter.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Counter } from './Counter';

describe('Counter', () => {
  it('초기 카운트 0을 보여준다', () => {
    render(<Counter />);
    expect(screen.getByText('카운터: 0')).toBeInTheDocument();
  });

  it('+1 클릭 시 카운트가 증가한다', async () => {
    render(<Counter />);
    await userEvent.click(screen.getByRole('button', { name: '+1' }));
    expect(screen.getByText('카운터: 1')).toBeInTheDocument();
  });

  it('-1 클릭 시 카운트가 감소한다', async () => {
    render(<Counter />);
    await userEvent.click(screen.getByRole('button', { name: '-1' }));
    expect(screen.getByText('카운터: -1')).toBeInTheDocument();
  });
});
```

### useQuery가 있는 컴포넌트 테스트 (MSW)

```tsx
// src/test/msw/handlers.ts
import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('/api/todos', () => {
    return HttpResponse.json([
      { id: 1, title: '테스트 할 일', completed: false },
    ]);
  }),
];
```

```tsx
// TodoList.test.tsx
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { server } from '@/test/msw/server';
import { TodoList } from './TodoList';

// MSW 서버 설정
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

function renderWithQuery(component: React.ReactNode) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>,
  );
}

it('할 일 목록을 불러와서 보여준다', async () => {
  renderWithQuery(<TodoList />);

  // 로딩 상태 확인
  expect(screen.getByText('로딩 중...')).toBeInTheDocument();

  // 데이터 로드 후 확인
  expect(await screen.findByText('테스트 할 일')).toBeInTheDocument();
});
```

## 무엇을 테스트하지 않는가

- 구현 내부 로직 (내부 state 직접 확인 금지)
- CSS 스타일 값
- 라이브러리 자체 동작 (React, Next.js는 이미 테스트됨)
- `console.log` 출력

**이유**: 구현 세부사항을 테스트하면 리팩토링할 때마다 테스트가 깨진다. 사용자가 보는 결과를 테스트해야 코드를 자유롭게 개선할 수 있다.

## ESLint + Prettier + Husky 설정

### Husky + lint-staged

```bash
npm install -D husky lint-staged
npx husky init
```

```json
// package.json
{
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{json,md,css}": ["prettier --write"]
  }
}
```

```bash
# .husky/pre-commit
npx lint-staged
```

## 테스트 설명 방식

처음 테스트를 작성하는 학습자에게는 다음 질문으로 테스트 설계를 돕는다:

```
1. 이 컴포넌트/함수를 사용하는 "사람"은 무엇을 기대하는가?
2. 어떤 상황(입력)에서, 어떤 결과(출력/화면)가 나와야 하는가?
3. 이것이 망가지면 사용자가 어떤 불편을 겪는가?
```
