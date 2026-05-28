# 커스텀 훅(Custom Hooks)

## 1. 문제 제기 — 커스텀 훅이 없으면?

여러 컴포넌트에서 똑같은 로직이 반복됩니다.

예를 들어 "로그인 폼"과 "회원가입 폼" 두 컴포넌트가 모두 `localStorage`에 값을 저장한다고 해봅시다.

```tsx
// LoginForm.tsx
const [email, setEmail] = useState(() => localStorage.getItem('email') ?? '');
useEffect(() => {
  localStorage.setItem('email', email);
}, [email]);

// SignupForm.tsx  ← 똑같은 코드를 또 씁니다
const [email, setEmail] = useState(() => localStorage.getItem('email') ?? '');
useEffect(() => {
  localStorage.setItem('email', email);
}, [email]);
```

같은 코드가 두 군데에 복제되어 있습니다. 만약 `localStorage`를 `sessionStorage`로 바꿔야 한다면?
두 파일 모두 고쳐야 합니다. 컴포넌트가 10개면 10번 고쳐야 합니다.

**JS에서 반복되는 코드를 함수로 빼는 것**처럼, **React에서 반복되는 훅 로직을 빼는 도구가 커스텀 훅**입니다.

---

## 2. 커스텀 훅이란?

> **커스텀 훅 = `useState`, `useEffect` 같은 훅을 내부에서 사용하는 "재사용 가능한 함수"**

비유: 평범한 JS 함수는 "값을 받아 값을 돌려주는 기계"입니다. 커스텀 훅은
"React 상태와 생명주기를 다룰 줄 아는 특별한 함수"입니다.

```tsx
// 평범한 JS 함수
function add(a, b) {
  return a + b;
}

// 커스텀 훅 (이름이 use로 시작!)
function useCounter() {
  const [count, setCount] = useState(0);
  const increase = () => setCount((c) => c + 1);
  return { count, increase };
}
```

사용할 때는 다른 훅과 똑같이 컴포넌트 안에서 호출합니다.

```tsx
function MyComponent() {
  const { count, increase } = useCounter();
  return <button onClick={increase}>{count}</button>;
}
```

---

## 3. ⚠️ `use` prefix 규칙 — 왜 이름이 `use`로 시작해야 하나?

React는 함수 이름이 **`use`로 시작하면 "이건 훅이구나"** 라고 인식합니다.

이것이 중요한 이유:

1. **React가 훅 규칙(Rules of Hooks)을 검사**합니다.
   - 훅은 컴포넌트 최상위에서만 호출 가능, 조건문/반복문 안에서 호출 금지.
   - 이 검사는 `use`로 시작하는 함수에만 적용됩니다.
2. **ESLint 플러그인(`eslint-plugin-react-hooks`)이 자동으로 검증**합니다.

```tsx
// 좋음 — React가 훅으로 인식
function useLocalStorage() { ... }

// 나쁨 — React는 그냥 함수로 봅니다. 내부에서 useState를 쓰면 에러!
function localStorage() {
  const [v, setV] = useState(''); // 규칙 위반 감지 안 됨
}
```

**핵심**: 내부에서 다른 훅을 호출한다면 반드시 `use`로 시작하라.

---

## 4. `useLocalStorage` 동작 원리

목표: `useState`처럼 쓰지만, 값이 자동으로 `localStorage`에 저장되어 새로고침해도 유지되는 훅.

```tsx
function useLocalStorage(key: string, initialValue: string) {
  // 1. 초기값: localStorage에 저장된 값이 있으면 그것, 없으면 initialValue
  const [value, setValue] = useState(() => {
    const stored = localStorage.getItem(key);
    return stored ?? initialValue;
  });

  // 2. value가 바뀔 때마다 localStorage에 동기화
  useEffect(() => {
    localStorage.setItem(key, value);
  }, [key, value]);

  // 3. useState처럼 [값, 설정함수] 반환
  return [value, setValue] as const;
}
```

사용:

```tsx
const [email, setEmail] = useLocalStorage('email', '');
// useState와 사용법이 완전히 똑같음. 새로고침해도 email은 유지됨.
```

**포인트**: 내부는 `useState` + `useEffect`의 조합. 외부에는 `useState`처럼 보이게 만든 것.

---

## 5. `useToggle` 동작 원리

목표: true/false를 토글하는 흔한 패턴을 한 줄로.

```tsx
function useToggle(initial = false) {
  const [value, setValue] = useState(initial);
  const toggle = () => setValue((v) => !v);
  return [value, toggle] as const;
}
```

사용:

```tsx
const [isOpen, toggleOpen] = useToggle(false);

return <button onClick={toggleOpen}>{isOpen ? '닫기' : '열기'}</button>;
```

**비유**: 전등 스위치를 위한 전용 리모컨. 매번 `setValue(!value)`를 쓰는 대신 `toggle()` 한 번이면 끝.

---

## 6. `useFetch` 간략 소개 (다음 단계 예고)

API 호출(서버에서 데이터 가져오기)도 컴포넌트마다 똑같이 반복됩니다.

```tsx
function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setLoading(true);
    fetch(url)
      .then((res) => res.json())
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [url]);

  return { data, loading, error };
}
```

사용:

```tsx
const { data, loading, error } = useFetch<User[]>('/api/users');
```

> 다음 단계에서는 이 `useFetch`의 한계(캐싱 없음, 중복 요청, 에러 재시도 등)를 알아보고, 그 해결책으로 **TanStack Query**를 학습합니다.

---

## 7. Best Case — 커스텀 훅이 빛나는 순간

| 상황 | 이유 |
|------|------|
| **같은 훅 로직이 2개 이상 컴포넌트에서 반복** | 한 곳에서 고치면 전체 적용 |
| **컴포넌트가 너무 길어서 읽기 힘들 때** | 관심사 분리. 화면(JSX)과 상태 로직을 나눔 |
| **테스트하기 어려운 로직을 분리하고 싶을 때** | 훅 단독으로 테스트 가능 |
| **외부 시스템(localStorage, WebSocket, API)과의 연동** | 컴포넌트는 "어떻게 가져오는지"를 몰라도 됨 |

예: 5개 페이지에서 모두 다크모드 토글이 필요 → `useDarkMode` 훅 하나로 해결.

---

## 8. Worst Case — 커스텀 훅을 잘못 쓰면

| 상황 | 문제 |
|------|------|
| **한 번만 쓰이는 로직을 훅으로 분리** | 추상화 비용만 늘고 이득 없음. 컴포넌트 안에 두는 게 낫다 |
| **훅 안에서 조건부로 다른 훅 호출** | Rules of Hooks 위반. 렌더링 순서가 깨져 버그 발생 |
| **`use`로 시작하지 않는 이름 사용** | ESLint가 검증 못 함. 잘못된 훅 사용을 놓침 |
| **너무 많은 책임을 한 훅에 몰아넣음** | `useEverything()` 같은 거대 훅은 디버깅 지옥 |
| **컴포넌트 외부(일반 함수)에서 호출** | 훅은 컴포넌트나 다른 훅 안에서만 호출 가능 |

⚠️ 잘못된 예:

```tsx
function useBad(flag: boolean) {
  if (flag) {
    const [x] = useState(0); // 조건부 훅 호출 — 금지!
  }
}
```

---

## 9. 흔한 실수 모음

1. **이름을 `getLocalStorage`로 짓기** → React가 훅으로 인식 못 함. 반드시 `use`로 시작.
2. **반환값 형태를 헷갈리기** → `useState`처럼 배열로 반환할지, 객체로 반환할지 일관성 있게 정하기.
   - 값이 2개고 흔하게 비구조화 → 배열 `[v, set]`
   - 값이 3개 이상이거나 의미가 다양 → 객체 `{ data, loading, error }`
3. **`useEffect` 의존성 배열에 `key` 누락** → 외부에서 `key`가 바뀌어도 동기화 안 됨.
4. **SSR(Next.js) 환경에서 `localStorage` 직접 접근** → 서버에는 `window`가 없어서 에러. 가드 필요.

---

## 10. 이해 확인 질문

1. 아래 함수는 커스텀 훅으로 동작할까요? 안 된다면 왜 그럴까요?
   ```tsx
   function getCounter() {
     const [count, setCount] = useState(0);
     return count;
   }
   ```

2. `useLocalStorage('user', '')`를 두 컴포넌트에서 동시에 호출했을 때, 한 컴포넌트에서 값을 바꾸면 다른 컴포넌트의 값도 자동으로 바뀔까요? 안 바뀐다면 왜일까요?

3. `useToggle`을 `useState` 없이 만들 수 있을까요? 만들 수 없다면 그 이유는?

---

## 심화 Q&A

(예제 학습 중 나온 질문은 이 아래에 누적 기록)
