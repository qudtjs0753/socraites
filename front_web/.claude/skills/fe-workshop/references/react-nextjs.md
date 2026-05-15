# React + Next.js 패턴 레퍼런스

## 목차

1. [컴포넌트 설계 원칙](#컴포넌트-설계)
2. [Custom Hook 패턴](#custom-hook)
3. [Next.js App Router 패턴](#app-router)
4. [Todo 앱 예제 (전체 흐름)](#todo-예제)

---

## 컴포넌트 설계 원칙 {#컴포넌트-설계}

### 언제 컴포넌트를 분리하는가

```
1. 같은 UI가 2번 이상 반복되면 → 컴포넌트 추출
2. 하나의 컴포넌트가 100줄 이상이면 → 분리 고려
3. 로직과 UI가 섞여서 읽기 어려우면 → 훅으로 로직 분리
```

### Server vs Client 결정 규칙

```
useState, useEffect, onClick, onChange 사용?
├── Yes → 'use client' 추가
└── No  → Server Component (기본값, 아무것도 추가 안 해도 됨)

DB, 환경 변수(API_KEY) 직접 접근?
└── Server Component만 가능 (Client에서 접근하면 보안 위험)
```

### Props 타입 정의

```tsx
// 나쁜 예: any 사용
function Card(props: any) { ... }

// 좋은 예: 명시적 타입
interface CardProps {
  title: string;
  content: string;
  imageUrl?: string;    // ? = 선택 사항
  onClick?: () => void;
}

function Card({ title, content, imageUrl, onClick }: CardProps) {
  return (
    <div onClick={onClick} className="...">
      <h2>{title}</h2>
      <p>{content}</p>
      {imageUrl && <img src={imageUrl} alt={title} />}
    </div>
  );
}
```

---

## Custom Hook 패턴 {#custom-hook}

### 기본 패턴

```tsx
// hooks/useLocalStorage.ts
function useLocalStorage<T>(key: string, initialValue: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setStoredValue = (newValue: T) => {
    setValue(newValue);
    localStorage.setItem(key, JSON.stringify(newValue));
  };

  return [value, setStoredValue] as const;
}

// 사용
const [theme, setTheme] = useLocalStorage('theme', 'light');
```

### useDebounce (검색 최적화)

```tsx
// hooks/useDebounce.ts
function useDebounce<T>(value: T, delay: number = 300): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// 사용: 검색어가 300ms 동안 안 바뀌면 API 호출
const debouncedSearch = useDebounce(searchQuery, 300);
useEffect(() => {
  if (debouncedSearch) fetchResults(debouncedSearch);
}, [debouncedSearch]);
```

---

## Next.js App Router 패턴 {#app-router}

### 폴더 구조와 URL

```
app/
├── page.tsx              → /
├── about/
│   └── page.tsx          → /about
├── blog/
│   ├── page.tsx          → /blog
│   └── [slug]/
│       └── page.tsx      → /blog/react-hooks (동적)
└── dashboard/
    ├── layout.tsx        → /dashboard/** 공통 레이아웃
    └── page.tsx          → /dashboard
```

### 서버에서 데이터 가져오기 (Server Component)

```tsx
// app/blog/page.tsx (Server Component - 'use client' 없음)
async function BlogPage() {
  // 이 fetch는 서버에서 실행됨
  const posts = await fetch('https://api.example.com/posts', {
    next: { revalidate: 3600 }, // ISR: 1시간마다 갱신
  }).then(res => res.json());

  return (
    <div>
      {posts.map(post => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
}
```

### loading.tsx + error.tsx

```tsx
// app/blog/loading.tsx
export default function Loading() {
  return <div className="animate-pulse">로딩 중...</div>;
}

// app/blog/error.tsx
'use client';
export default function Error({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <div>
      <p>오류가 발생했습니다: {error.message}</p>
      <button onClick={reset}>다시 시도</button>
    </div>
  );
}
```

---

## Todo 앱 예제 (전체 흐름) {#todo-예제}

JSONPlaceholder API 사용 (실제 서버 없이 연습 가능).

### API 함수

```tsx
// lib/api.ts
const API_URL = 'https://jsonplaceholder.typicode.com';

export interface Todo {
  id: number;
  title: string;
  completed: boolean;
  userId: number;
}

export const todoApi = {
  getAll: (): Promise<Todo[]> =>
    fetch(`${API_URL}/todos?_limit=10`).then(res => res.json()),

  create: (title: string): Promise<Todo> =>
    fetch(`${API_URL}/todos`, {
      method: 'POST',
      body: JSON.stringify({ title, completed: false, userId: 1 }),
      headers: { 'Content-Type': 'application/json' },
    }).then(res => res.json()),

  toggle: (id: number, completed: boolean): Promise<Todo> =>
    fetch(`${API_URL}/todos/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ completed }),
      headers: { 'Content-Type': 'application/json' },
    }).then(res => res.json()),

  delete: (id: number): Promise<void> =>
    fetch(`${API_URL}/todos/${id}`, { method: 'DELETE' }).then(() => {}),
};
```

### TodoList 컴포넌트

```tsx
// components/todo/TodoList.tsx
'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { todoApi } from '@/lib/api';

export function TodoList() {
  const queryClient = useQueryClient();

  const { data: todos, isLoading, error } = useQuery({
    queryKey: ['todos'],
    queryFn: todoApi.getAll,
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, completed }: { id: number; completed: boolean }) =>
      todoApi.toggle(id, completed),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['todos'] });
    },
  });

  if (isLoading) return <p>로딩 중...</p>;
  if (error) return <p>오류가 발생했습니다.</p>;

  return (
    <ul className="space-y-2">
      {todos?.map(todo => (
        <li
          key={todo.id}
          className="flex items-center gap-2 rounded border p-3"
        >
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={() =>
              toggleMutation.mutate({ id: todo.id, completed: !todo.completed })
            }
          />
          <span className={todo.completed ? 'line-through text-gray-400' : ''}>
            {todo.title}
          </span>
        </li>
      ))}
    </ul>
  );
}
```
