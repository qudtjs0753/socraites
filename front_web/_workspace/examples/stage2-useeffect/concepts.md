# useEffect 개념 정리

## 사전 지식 — setInterval

`setInterval`은 **일정 시간마다 코드를 반복 실행해줘** 를 브라우저에게 부탁하는 함수다.

```js
setInterval(() => {
  console.log('안녕');
}, 1000); // 1000ms = 1초마다
```

```
1초 뒤 → "안녕"
2초 뒤 → "안녕"
3초 뒤 → "안녕"
... 멈추라고 할 때까지 계속
```

**멈추려면 — clearInterval**

```js
const id = setInterval(() => {
  console.log('안녕');
}, 1000);

clearInterval(id); // 등록할 때 받은 id로 해당 타이머만 정지
```

`setInterval`은 등록할 때 **번호표(id)** 를 반환한다. 나중에 그 번호표로 어떤 타이머를 멈출지 특정할 수 있다.

**⚠️ 비동기 — 기다리지 않는다**

```js
console.log('A');

setInterval(() => {
  console.log('반복');
}, 1000);

console.log('B'); // setInterval을 기다리지 않고 즉시 실행
```

```
A    ← 즉시
B    ← 즉시
반복 ← 1초 뒤
반복 ← 2초 뒤
...
```

`setInterval`은 "1초마다 실행해줘" 를 브라우저에 예약만 하고 **즉시 다음 줄로 넘어간다.** 1초를 기다리지 않는다.

---

## 왜 필요한가?

React 컴포넌트는 **화면을 그리는 것**이 본업이다.
그런데 개발하다 보면 화면 그리기 외에도 해야 할 일이 생긴다.

- 페이지가 열리면 API를 호출해서 데이터를 가져오고 싶다
- 타이머를 시작하고 싶다
- 브라우저 창 크기 변화를 감지하고 싶다

이런 "화면 그리기 외의 일"을 **부수 효과(side effect)** 라고 부른다.
`useEffect`는 이 부수 효과를 **렌더링이 끝난 다음에 실행**하도록 등록하는 Hook이다.

> 💡 비유: 식당에서 "음식을 테이블에 놓은 다음에 물 가져다주기"처럼,
> 화면이 그려진 이후에 해야 할 일을 등록하는 것.

---

## 기본 형태

```tsx
useEffect(() => {
  // 렌더링 후 실행할 코드
}, [의존성]);
```

의존성 배열에 따라 실행 시점이 달라진다.

| 의존성 배열 | 실행 시점 |
|------------|----------|
| 없음 (생략) | 매 렌더링 후마다 실행 |
| `[]` (빈 배열) | 컴포넌트가 처음 나타날 때 **1회만** 실행 |
| `[값]` | 컴포넌트가 처음 나타날 때 + **값이 바뀔 때마다** 실행 |

```tsx
// ✅ 처음 한 번만 실행 (빈 배열)
useEffect(() => {
  console.log('마운트됨!');
}, []);

// ✅ userId가 바뀔 때마다 실행
useEffect(() => {
  fetchUser(userId);
}, [userId]);
```

---

## 흔한 실수

**실수 1: 의존성 배열을 아예 빠뜨리기**
```tsx
// ❌ 렌더링마다 API 호출됨 → 무한 루프 위험
useEffect(() => {
  fetchData();
});

// ✅ 처음 한 번만 호출
useEffect(() => {
  fetchData();
}, []);
```

**실수 2: 의존성에 써야 할 값을 빠뜨리기**
```tsx
// ❌ id가 바뀌어도 재실행 안 됨
useEffect(() => {
  fetchUser(id);
}, []);

// ✅ id를 의존성에 포함
useEffect(() => {
  fetchUser(id);
}, [id]);
```

> ⚠️ ESLint의 `react-hooks/exhaustive-deps` 규칙이 이 실수를 잡아준다.

---

## 심화 — cleanup 함수

`useEffect`가 함수를 반환하면, 그 함수는 **다음 effect 실행 직전** 또는 **컴포넌트가 사라질 때** 호출된다.
타이머, 이벤트 리스너처럼 "등록한 것을 나중에 해제"해야 할 때 사용한다.

```tsx
useEffect(() => {
  if (!running) return;
  const id = setInterval(() => {
    setElapsed((prev) => prev + 10);
  }, 10);
  return () => clearInterval(id); // ← cleanup: 정지 시 / 언마운트 시 정리
}, [running]);
```

WHY: `running`이 `true → false`로 바뀌면 effect가 재실행되기 전에 cleanup이 먼저 호출된다. 이 덕분에 이전 interval이 깔끔히 제거된다.

---

## 심화 — setInterval + useEffect

interval은 외부 시스템(브라우저 타이머)이라 React 상태 흐름 바깥에 있다.
`useEffect`로 감싸야 컴포넌트 생명주기와 동기화할 수 있다.

```tsx
useEffect(() => {
  if (!running) return;

  const id = setInterval(() => {
    setElapsed((prev) => prev + 10); // ← 10ms마다 10씩 증가
  }, 10); // ← 두 번째 인자 10은 "10ms 간격"을 의미 (의존성 배열이 아님)

  return () => clearInterval(id); // ← cleanup: running이 바뀌거나 언마운트 시 정리
}, [running]); // ← running이 바뀔 때마다 재실행
```

> `setInterval(콜백, 간격ms)` — 두 번째 인자는 실행 간격이다. `[running]` 이 의존성 배열이다.

**⚠️ 흔한 실수: 함수형 업데이트를 쓰지 않는 경우**

`setInterval`은 브라우저가 관리하는 타이머다. 콜백 안에서 `setState`를 호출하면 **React 바깥 세계가 React 안으로 신호를 보내는 것**이다.

```
브라우저 타이머 (React 바깥)
  → 10ms마다 콜백 실행
  → setElapsed(...) 호출  ← React 안으로 진입
  → React: "state 바뀌었네, 리렌더!"
```

문제는 바깥 세계(타이머 콜백)가 React의 최신 state를 모를 수 있다는 것이다.

```tsx
// ❌ elapsed를 직접 참조
setInterval(() => {
  setElapsed(elapsed + 10); // effect가 처음 만들어질 때의 elapsed(=0)를 계속 참조
}, 10);

// ✅ 함수형 업데이트
setInterval(() => {
  setElapsed((prev) => prev + 10); // React에게 "최신 값 줘, 거기서 계산할게" 요청
}, 10);
```

`elapsed`를 직접 쓰면 클로저(closure) 때문에 effect가 만들어질 당시의 값이 고정된다.
`prev =>` 형태를 쓰면 React가 항상 최신 state를 인자로 넘겨준다.

---

## 심화 — window 이벤트 리스너 + useEffect

`window`는 React 바깥의 전역이다. 등록과 해제를 짝맞춰야 한다.

```tsx
useEffect(() => {
  const handleResize = () => setWidth(window.innerWidth);
  handleResize();
  window.addEventListener('resize', handleResize); // ← 등록
  return () => window.removeEventListener('resize', handleResize); // ← 해제
}, []);
```

WHY: 빈 배열 `[]`은 "마운트 시 1회 등록, 언마운트 시 1회 해제"를 의미한다. cleanup을 빠뜨리면 컴포넌트가 사라져도 리스너가 남아 메모리 누수가 생긴다.

---

## 의존성 배열 패턴 요약

| 패턴 | 언제 씀 |
|------|--------|
| `[running]` | running이 바뀔 때마다 effect 재실행 (새 interval 시작, 이전 건 cleanup으로 정리) |
| `[]` | 1회만 등록 (resize 리스너처럼 상태와 무관한 외부 이벤트) |

> 의존성 누락 시 ESLint(`react-hooks/exhaustive-deps`)가 경고를 띄운다.

---

## 심화 — 실행 순서: useState 초기화와 useEffect의 관계

"초기 렌더링 이후에 실행된다"는 말은 정확히 어떤 순서인가?

```
1. useState 초기화        ← 동기(sync), 렌더링 중 실행
2. 컴포넌트 함수 실행     ← JSX 계산
3. DOM에 반영            ← 화면에 그리기
4. useEffect 실행        ← 여기가 "렌더링 이후"
```

`useState(0)` 은 컴포넌트 함수가 실행될 때 동기적으로 처리된다.
`useEffect` 가 실행될 때는 이미 `useState` 초기화가 완료된 상태다.

```tsx
function Counter() {
  const [count, setCount] = useState(0); // 1️⃣ 렌더링 중 초기화

  useEffect(() => {
    console.log('현재 count:', count); // 3️⃣ DOM 반영 이후 실행 → 0 출력
  }, []);

  return <div>{count}</div>; // 2️⃣ JSX → DOM 반영
}
```

---

## ⚠️ 심화 — useEffect 안에서 setState 시 렌더링 횟수

`useEffect` 안에서 `setState`를 호출하면 **렌더링이 2번** 일어난다.

```tsx
useEffect(() => {
  setCount(1); // state 변경 → 리렌더 예약
}, []);
```

```
렌더 1 (count=0) → DOM 반영 → useEffect → setCount(1)
렌더 2 (count=1) → DOM 반영 → useEffect 실행 안 함 ([] 이므로)
```

원시값(`1`, `"hello"`)은 같은 값을 다시 set하면 React가 `Object.is` 로 비교해 **리렌더를 중단**한다.

---

## ⚠️ 심화 — 참조값(배열·객체) setState의 무한루프

배열·객체는 내용이 같아도 매번 새로운 참조가 생성된다.
의존성 배열 없이 `setState([])`를 호출하면 **무한루프**가 발생한다.

```tsx
// 💥 무한루프
useEffect(() => {
  setList([]); // 매번 새 [] 참조 → Object.is([], []) === false → 리렌더 → 반복
});
```

```
Object.is(1, 1)   // ✅ true  → 같음 → 렌더 중단
Object.is([], []) // ❌ false → 다름 → 리렌더 발생
```

| 타입 | 예시 | 무한루프 위험 |
|------|------|-------------|
| 원시값 | `1`, `"hello"`, `true` | ❌ |
| 배열 | `[]`, `[1,2,3]` | ✅ |
| 객체 | `{}`, `{a:1}` | ✅ |

**해결책**: 의존성 배열로 실행 조건을 반드시 통제한다.

```tsx
// ✅ 안전 — 마운트 시 1회만
useEffect(() => {
  setList([]);
}, []);

// ✅ 안전 — API 응답을 담는 경우
useEffect(() => {
  fetchItems().then(data => setList(data));
}, []);
```

> **규칙**: `useEffect`를 쓸 때는 항상 `[]` 또는 `[변수]`를 명시한다.
> "의존성 배열 없는 useEffect"는 원칙적으로 사용하지 않는다.

---

## useEffect Best Case vs Worst Case

### ✅ Best Case — useEffect가 적합한 상황

외부 시스템(API, 브라우저, 타이머)과 연결할 때 사용한다.
-> 내 해석: React가 주도할 수 없는 외부의 lifecycle을 조작하려 할 때.

```tsx
// ✅ 마운트 시 데이터 패칭
useEffect(() => {
  fetchUser(userId).then(data => setUser(data));
}, [userId]);

// ✅ 타이머 등록 + cleanup
useEffect(() => {
  const id = setInterval(() => setTime(t => t + 1), 1000);
  return () => clearInterval(id);
}, []);

// ✅ 이벤트 리스너 등록 + cleanup
useEffect(() => {
  window.addEventListener('resize', handleResize);
  return () => window.removeEventListener('resize', handleResize);
}, []);
```

### 💥 Worst Case — useEffect 없이 해결할 수 있는 것들

**1. 다른 state에서 파생되는 값 계산**

```tsx
// ❌ 불필요한 useEffect → 렌더가 2번 일어남
useEffect(() => {
  setCount(items.length);
}, [items]);

// ✅ 렌더링 중에 바로 계산
const count = items.length;
```

**2. 이벤트 핸들러로 처리할 수 있는 것**

```tsx
// ❌ 클릭 결과를 useEffect로 처리
useEffect(() => {
  if (clicked) sendLog();
}, [clicked]);

// ✅ 이벤트 핸들러에서 직접 처리
function handleClick() {
  sendLog();
  setClicked(true);
}
```

**3. props 변화에 따른 state 초기화**

```tsx
// ❌ useEffect로 초기화 → 렌더 2번
useEffect(() => {
  setValue('');
}, [userId]);

// ✅ key prop으로 컴포넌트 자체를 리셋
<Profile key={userId} userId={userId} />
```

### 한 줄 판단 기준

> **"React 바깥 시스템(API, 브라우저, 타이머)과 연결하는 작업인가?"**
> → Yes → useEffect 적합
> → No → 렌더링 중 계산하거나 이벤트 핸들러로 처리
