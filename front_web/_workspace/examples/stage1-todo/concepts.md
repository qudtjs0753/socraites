# Stage 1 개념 설명: props · 리스트 렌더링 · 조건부 렌더링

## props — 컴포넌트 간 데이터 전달

### 문제
같은 모양의 항목을 여러 개 그려야 할 때, 컴포넌트를 한 번만 정의하고 데이터만 바꿔서 재사용하고 싶다.

### 핵심
props = 부모 컴포넌트가 자식 컴포넌트에 전달하는 데이터(또는 함수).

```tsx
// 자식: props를 받아서 화면을 그린다
function TodoItem({ text, completed }: { text: string; completed: boolean }) {
  return <div>{text}</div>;
}

// 부모: 데이터를 props로 넘긴다
<TodoItem text="공부하기" completed={false} />
<TodoItem text="운동하기" completed={true} />
```

### 중요 규칙
- props는 **부모 → 자식 방향으로만** 흐른다.
- 자식이 부모의 상태를 바꾸고 싶을 때는, **부모가 함수를 props로 내려주고** 자식이 그 함수를 호출한다.

```tsx
// 부모가 "삭제" 동작을 함수로 내려줌
<TodoItem onDelete={handleDelete} />

// 자식은 받은 함수를 호출만 함 (직접 상태 변경 불가)
<button onClick={() => onDelete(todo.id)}>삭제</button>
```

### 흔한 실수
자식에서 부모 상태를 직접 바꾸려는 것. props로 받은 함수를 호출하는 방식으로 해야 한다.

---

## 리스트 렌더링 — map + key

### 핵심
배열 데이터를 화면에 나열할 때 `map`을 사용한다.

```tsx
const todos = [
  { id: 1, text: '공부하기' },
  { id: 2, text: '운동하기' },
];

// 배열 → JSX 변환
todos.map(todo => (
  <li key={todo.id}>{todo.text}</li>
))
```

### key가 왜 필요한가?
React는 목록이 바뀌면 어떤 항목이 변경됐는지 파악해야 한다. `key`가 없으면 전체를 다시 그려서 느려지고, 버그가 생길 수 있다.

```tsx
// ❌ key 없음 → 콘솔 경고 + 잠재적 버그
todos.map(todo => <li>{todo.text}</li>)

// ✅ 고유한 key
todos.map(todo => <li key={todo.id}>{todo.text}</li>)
```

**key 선택 기준**: 목록 안에서 절대 중복되지 않는 값. 보통 DB의 `id`를 사용한다. 배열 인덱스(`index`)는 항목 순서가 바뀌면 버그가 생기므로 권장하지 않는다.

---

## 조건부 렌더링 — 상황에 따라 다른 UI

### 패턴 1: && (앞이 true일 때만 보여주기)

```tsx
// todos가 비어있을 때만 메시지 표시
{todos.length === 0 && <p>할 일이 없어요!</p>}
```

### 패턴 2: 삼항 연산자 (둘 중 하나 선택)

```tsx
// completed면 취소선, 아니면 일반 텍스트
<span className={completed ? 'line-through text-gray-400' : 'text-gray-800'}>
  {text}
</span>
```

### 패턴 3: 일찍 반환 (컴포넌트 전체를 조건 처리)

```tsx
function MyComponent({ isLoading, data }) {
  if (isLoading) return <p>로딩 중...</p>;   // 여기서 함수 종료
  if (!data)     return <p>데이터 없음</p>;  // 여기서 함수 종료
  return <실제_컨텐츠 data={data} />;
}
```

---

## Todo 앱 코드 읽기

`page.tsx`를 열면 이 구조로 되어 있다:

```
TodoPage (부모)
│
│ state: todos[], inputValue
│ 함수: handleAdd, handleToggle, handleDelete
│
├── <input> + <button> → 새 할 일 추가
│
├── 조건부 렌더링: todos.length === 0 → 빈 메시지
│
└── todos.map(todo => (         ← 리스트 렌더링
      <TodoItem                 ← 자식 컴포넌트
        key={todo.id}           ← 리스트 필수 key
        todo={todo}             ← 데이터 props
        onToggle={handleToggle} ← 함수 props
        onDelete={handleDelete} ← 함수 props
      />
    ))

TodoItem (자식)
│ props: todo, onToggle, onDelete
└── 체크박스 → onToggle(todo.id) 호출
└── 삭제 버튼 → onDelete(todo.id) 호출
```

### 이해 확인

1. `TodoItem` 안에서 `setTodos`를 직접 부르지 않는 이유는 무엇인가?
2. `key={todo.id}` 대신 `key={index}`를 쓰면 어떤 문제가 생길 수 있을까?
3. `{...todo, completed: !todo.completed}` 에서 `...todo`는 무엇을 하는가?

---

## 심화 Q&A

### ⚠️ `...todo`의 의미 — 몰랐던 개념

`handleToggle`의 `...todo`는 "나머지 todo를 제외한 것"이 **아니라**, **현재 `todo` 객체의 모든 속성을 복사**하는 스프레드 연산자예요.

```tsx
const todo = { id: 1, text: '공부하기', completed: false }

{ ...todo }
// → { id: 1, text: '공부하기', completed: false }  // 그대로 복사

{ ...todo, completed: true }
// → { id: 1, text: '공부하기', completed: true }   // completed만 덮어쓰기
```

"전체 복사 + 일부만 덮어쓰기" 패턴. 엑셀 행 전체를 복사한 뒤 한 칸만 수정하는 것과 같음.

---

### `handleToggle`에서 `map`을 쓰는 이유 — React 불변성 규칙

`map`이 전체 배열을 순회하는 건 맞음. 더 효율적인 방법도 있지만, React에서 `map`을 쓰는 이유가 있음.

**React는 state를 직접 수정하는 것을 금지한다:**

```tsx
// ❌ React가 변경을 감지 못함
todos[0].completed = true

// ✅ 새 배열을 만들어야 React가 감지하고 화면을 다시 그림
setTodos(todos.map(...))
```

**대안 비교:**

| 방법 | 성능 | 코드 가독성 | 언제 쓰나 |
|------|------|------------|----------|
| `map` (현재) | O(n) | ★★★ 좋음 | 대부분의 경우 |
| `findIndex` + 배열 복사 | O(n) | ★★ 보통 | 거의 안 씀 |
| 객체(Record)로 저장 | O(1) | ★ 복잡함 | 수천 개 이상의 대용량 목록 |

→ Todo 앱처럼 항목이 적으면 `map`이 정답. 객체 방식은 채팅 메시지, 대용량 테이블 등에서 씀.

---

### ⚠️ React가 state 변화를 감지하는 방법 — 몰랐던 개념

React는 `Object.is(이전값, 새값)`으로 상태 변화를 감지한다. 이는 사실상 `===` 비교로, **내용이 아니라 참조(메모리 주소)**를 비교한다.

```tsx
const arr = [{ id: 1, completed: false }]

// 직접 수정 → 같은 배열 객체
arr[0].completed = true
arr === arr  // true → React: "변화 없음" → 화면 안 바뀜

// 새 배열 생성
const newArr = arr.map(...)
arr === newArr  // false → React: "변화 감지" → 재렌더링
```

같은 내용의 객체라도 새로 만들면 주소가 다르다:

```tsx
{ completed: true } === { completed: true }  // false (다른 객체)
```

**핵심:** React가 아는 것은 "주소가 바뀌었냐"지, "내용이 바뀌었냐"가 아니다.
그래서 `map`, 스프레드(`[...todos]`, `{...todo}`)로 **새 객체를 만들어야** React가 변화를 인식하고 화면을 다시 그린다.

---

### ⚠️ 자식이 부모 state를 직접 못 바꾸는 이유 — 몰랐던 개념

**잘못된 이해:** "props로 넘길 때 객체가 복사되니까 수정해도 부모에 안 반영된다"

**실제:** 객체/배열은 props로 넘길 때 **복사가 아니라 같은 참조(주소)**가 넘어간다.

진짜 이유는 두 가지:
1. **`setTodos`가 자식 스코프에 없다** — setter는 부모에만 있어서 자식이 호출할 수 없다
2. **직접 수정하면 React가 감지 못한다** — 참조가 안 바뀌기 때문 (위 Q&A 참조)

```tsx
// 자식이 직접 수정 → 기술적으로 되긴 하지만
todo.completed = true  // 참조 그대로 → React 감지 못함 → 화면 안 바뀜
```

---

### ⚠️ 함수를 넘겨도 내부에서 직접 수정하면 감지 못함 — 몰랐던 개념

**핵심:** 누가 호출하든 상관없이, React가 보는 건 `setTodos`가 호출됐냐 아니냐다.

```tsx
// ❌ 부모가 이런 함수를 넘기면
function handleToggle(id) {
  todos[id].completed = true  // setTodos 없음 → React 감지 못함
}

// ✅ 이래야 동작
function handleToggle(id) {
  setTodos(todos.map(...))  // 새 배열 + setTodos 호출 → React 감지
}
```

**React 재렌더링 조건 (이 체인이 전부 충족돼야 한다):**

```
setTodos(새_값) 호출
    └── 새_값이 이전 참조와 다름
        └── React 감지 → 재렌더링
```

---

### setTodos를 자식에 직접 넘겨도 되지만 안 쓰는 이유

`setTodos`도 함수이므로 props로 넘길 수 있다. 하지만:

| | `setTodos` 직접 전달 | handler 함수 전달 |
|---|---|---|
| 자식이 아는 것 | 배열 전체 수정 로직 | "토글만 해줘" |
| 실수 가능성 | `setTodos([])` 로 전체 삭제 가능 | 의도한 동작만 가능 |
| 코드 중복 | 같은 로직이 여러 자식에 퍼짐 | 부모에 한 곳만 |

비유: 집 열쇠 전체를 주는 것(`setTodos`) vs 특정 방만 열리는 카드를 주는 것(`handleToggle`).

---

### ⚠️ Virtual DOM과 재렌더링 흐름 — 몰랐던 개념

React는 실제 DOM을 직접 건드리지 않는다. **Virtual DOM(가상 DOM)**이라는 JS 객체를 중간에 두고, 변경된 부분만 실제 DOM에 반영한다.

```
[컴포넌트 함수]
    │ JSX 반환
    ▼
[Virtual DOM]  ← JS 객체로 만든 DOM 구조 복사본
    │ diffing (비교)
    ▼
[실제 DOM]
```

**전체 흐름:**

```
setTodos(새배열) 호출
    │
    ▼
① Object.is 비교
   "이 컴포넌트 함수를 다시 실행할까?"
   → 참조 다름 → 재실행 결정
    │
    ▼
② 컴포넌트 함수 재실행 → 새 JSX 반환
   ⚠️ "새 Virtual DOM 생성" = 전체 트리 X
      state가 바뀐 컴포넌트 + 그 자식들만 함수 재실행
    │
    ▼
③ 이전 Virtual DOM vs 새 Virtual DOM 비교 (diffing)
   "어떤 노드가 바뀌었냐?"
   리스트라면 key로 항목 식별
    │
    ▼
④ 차이가 나는 실제 DOM node만 업데이트
```

**참조 비교(①)는 DOM을 찾는 작업이 아니라**, 함수를 재실행할지 결정하는 문지기다.
실제 DOM 노드 비교는 ③에서 일어난다.

| 단계 | 질문 | 수단 |
|---|---|---|
| ① | 함수를 다시 실행해야 하나? | 참조 비교 (Object.is) |
| ② | 어디까지 재실행하나? | state 변경 컴포넌트 + 자식만 |
| ③ | 어떤 DOM 노드가 바뀌었나? | Virtual DOM diffing + key |
| ④ | 실제 DOM 변경 | 최소 업데이트 |

**왜 Virtual DOM을 쓰냐면:** 실제 DOM 조작은 느리다.
Virtual DOM은 JS 객체라 비교가 빠르고,
변경 사항을 모아서 실제 DOM에 한 번에 최소한으로 반영할 수 있다.

---

### ⚠️ "새 Virtual DOM 생성"은 전체 트리가 아니다 — 몰랐던 개념

`setTodos`가 호출되면 컴포넌트 트리 전체가 재실행되는 게 아니다.
**state가 바뀐 컴포넌트부터 그 자식들만** 함수가 다시 실행된다.

```
App                ← 재실행 안 함
└── TodoPage       ← setTodos 호출됨 → 여기부터 아래만 재실행
    ├── <input>    ← 재실행
    ├── <button>   ← 재실행
    └── TodoItem × N  ← 재실행
```

**"재실행"과 "실제 DOM 업데이트"는 다르다:**

- 재실행 = 함수가 돌아서 새 JSX를 만드는 것 (빠름, JS 연산)
- 실제 DOM 업데이트 = diffing 후 변경된 부분만 (느림, 최소화)

`TodoItem` 10개가 있어도 1개만 바뀌었다면:
- 함수는 10개 다 재실행 (Virtual DOM 생성)
- 실제 DOM은 1개만 업데이트 (diffing 결과)

| | 범위 |
|---|---|
| 함수 재실행 | state 변경 컴포넌트 + 자식들 |
| 실제 DOM 업데이트 | diffing으로 찾은 변경 부분만 |
| 부모/형제 컴포넌트 | 재실행 안 함 |
