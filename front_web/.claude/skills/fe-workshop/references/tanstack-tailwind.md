# TanStack Query + Tailwind CSS 패턴 레퍼런스

## 목차

1. [TanStack Query 고급 패턴](#tanstack-query)
2. [Tailwind CSS 실용 패턴](#tailwind)
3. [함께 사용하는 패턴](#함께-사용)

---

## TanStack Query 고급 패턴 {#tanstack-query}

### queryKey 설계 원칙

```tsx
// 계층 구조로 설계
['todos']                        // 모든 todo
['todos', { status: 'active' }] // 조건부 필터
['todos', 1]                     // 특정 todo (id=1)
['users', userId, 'todos']       // 특정 사용자의 todo
```

**이유**: 계층 구조로 설계하면 `invalidateQueries`로 관련 캐시를 한 번에 무효화할 수 있다.

```tsx
// 'todos'로 시작하는 모든 캐시 무효화
queryClient.invalidateQueries({ queryKey: ['todos'] });
```

### useMutation + 낙관적 업데이트

```tsx
const deleteMutation = useMutation({
  mutationFn: todoApi.delete,
  onMutate: async (deletedId) => {
    // 진행 중인 refetch 취소
    await queryClient.cancelQueries({ queryKey: ['todos'] });

    // 현재 데이터 저장 (롤백용)
    const previousTodos = queryClient.getQueryData<Todo[]>(['todos']);

    // 낙관적으로 UI 업데이트 (서버 응답 전)
    queryClient.setQueryData<Todo[]>(['todos'], old =>
      old?.filter(todo => todo.id !== deletedId) ?? [],
    );

    return { previousTodos };
  },
  onError: (_error, _id, context) => {
    // 실패 시 롤백
    queryClient.setQueryData(['todos'], context?.previousTodos);
  },
  onSettled: () => {
    // 성공/실패 모두 refetch
    queryClient.invalidateQueries({ queryKey: ['todos'] });
  },
});
```

### 무한 스크롤

```tsx
import { useInfiniteQuery } from '@tanstack/react-query';

const { data, fetchNextPage, hasNextPage, isFetchingNextPage } = useInfiniteQuery({
  queryKey: ['posts'],
  queryFn: ({ pageParam = 1 }) =>
    fetch(`/api/posts?page=${pageParam}&limit=10`).then(res => res.json()),
  getNextPageParam: (lastPage, allPages) =>
    lastPage.hasMore ? allPages.length + 1 : undefined,
  initialPageParam: 1,
});

// 데이터 펼치기
const posts = data?.pages.flatMap(page => page.items) ?? [];
```

---

## Tailwind CSS 실용 패턴 {#tailwind}

### 재사용 컴포넌트 스타일링

```tsx
// lib/utils.ts (clsx + tailwind-merge 활용)
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// 사용: 충돌하는 클래스 자동 해결
cn('px-2 py-1', 'p-4')     // → 'p-4'
cn('text-red-500', isError && 'text-blue-500') // → 조건부 클래스
```

### Button 컴포넌트 (변형 지원)

```tsx
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

const variants = {
  primary: 'bg-blue-600 text-white hover:bg-blue-700',
  secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200',
  danger: 'bg-red-600 text-white hover:bg-red-700',
};

const sizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
};

export function Button({
  variant = 'primary',
  size = 'md',
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        'rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50',
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
}
```

### 반응형 레이아웃 패턴

```tsx
// 모바일 우선 (기본 → sm → md → lg 순서)
<div className="
  grid grid-cols-1     // 모바일: 1열
  sm:grid-cols-2       // 640px+: 2열
  lg:grid-cols-3       // 1024px+: 3열
  gap-4
">

// 반응형 네비게이션
<nav className="flex flex-col sm:flex-row items-start sm:items-center gap-2">

// 반응형 텍스트
<h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold">
```

### 로딩/스켈레톤 UI

```tsx
// 스켈레톤 컴포넌트
function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn('animate-pulse rounded bg-gray-200', className)} />
  );
}

// 카드 스켈레톤
function CardSkeleton() {
  return (
    <div className="rounded-lg border p-4 space-y-3">
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-2/3" />
    </div>
  );
}
```

---

## 함께 사용하는 패턴 {#함께-사용}

### 로딩/에러/성공 상태 UI 패턴

```tsx
function PostList() {
  const { data, isLoading, error, isError } = useQuery({
    queryKey: ['posts'],
    queryFn: fetchPosts,
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="rounded-lg bg-red-50 p-4 text-red-600">
        <p>오류: {error.message}</p>
        <button
          onClick={() => refetch()}
          className="mt-2 text-sm underline hover:no-underline"
        >
          다시 시도
        </button>
      </div>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {data?.map(post => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
}
```
