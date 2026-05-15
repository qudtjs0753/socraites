# Stage 1 개념 설명: React 첫 걸음

## JS 필수 문법 4가지

### ① 화살표 함수

React에서 함수를 짧게 쓰는 방식입니다.

```js
// 기존 방식
function add(a, b) {
  return a + b;
}

// 화살표 함수 (React에서 거의 이 방식 사용)
const add = (a, b) => a + b;

// 한 줄로 안 될 때
const greet = (name) => {
  const message = `안녕하세요, ${name}님!`;
  return message;
};
```

---

### ② 구조 분해 할당

객체/배열에서 원하는 값만 꺼내는 방법입니다.

```js
// 객체에서 꺼내기
const user = { name: '홍길동', age: 25 };
const { name, age } = user;  // name = '홍길동', age = 25

// React 컴포넌트 props에서 자주 이렇게 씀
function Card({ title, content }) {  // props 객체에서 바로 꺼냄
  return <div>{title}</div>;
}
```

---

### ③ 배열 map

배열의 모든 항목을 변환합니다. React 리스트 렌더링에 필수입니다.

```js
const fruits = ['사과', '바나나', '딸기'];

// 모든 과일을 화면에 나열하고 싶을 때
fruits.map(fruit => <li>{fruit}</li>)
// → <li>사과</li>, <li>바나나</li>, <li>딸기</li>
```

---

### ④ async/await

서버에서 데이터를 가져올 때 씁니다.

```js
// "await" = 결과가 올 때까지 기다려라
async function loadData() {
  const response = await fetch('https://api.example.com/data');
  const data = await response.json();
  return data;  // 서버 응답 데이터
}
```

> Stage 4(TanStack Query)에서 이 패턴을 훨씬 편하게 쓸 수 있게 됩니다.

---

## React는 왜 존재하는가?

### 문제: 순수 JS로 화면을 바꾸려면

```html
<p id="counter">0</p>
<button onclick="increment()">+1</button>

<script>
  let count = 0;
  function increment() {
    count++;
    document.getElementById('counter').innerText = count; // 직접 DOM 조작
  }
</script>
```

화면 요소가 수십~수백 개가 되면 "어떤 요소가 지금 어떤 상태인지" 추적하기 어려워지고, 요소들이 서로 영향을 주면 버그가 폭발적으로 늘어납니다.

### React의 핵심 아이디어

> **"상태(데이터)가 바뀌면 React가 알아서 화면을 다시 그린다."**

개발자는 "이 상태일 때 화면이 어떻게 보여야 하는가"만 선언하면 됩니다. DOM을 직접 조작할 필요가 없습니다.

---

## 컴포넌트와 JSX

**컴포넌트** = 화면의 부품을 만드는 함수  
**비유**: 붕어빵 틀 — 틀(컴포넌트) 하나로 다양한 붕어빵(인스턴스)을 만든다.

```tsx
// 컴포넌트 정의
function Card({ title, content }: { title: string; content: string }) {
  return (
    // JSX: JS 안에서 HTML처럼 쓰는 문법
    // 실제로는 JS 함수 호출로 변환됨
    <div>
      <h2>{title}</h2>      {/* {} 안에는 JS 표현식 */}
      <p>{content}</p>
    </div>
  );
}

// 같은 컴포넌트를 여러 번 재사용
<Card title="첫 번째" content="내용 1" />
<Card title="두 번째" content="내용 2" />
```

---

## useState

**useState** = 화면에 영향을 주는 변수를 만드는 방법

```tsx
const [count, setCount] = useState(0);
//     ↑ 현재 값   ↑ 값을 바꾸는 함수   ↑ 초기값
```

**핵심 규칙**: `count = count + 1`처럼 직접 수정하면 화면이 안 바뀐다.  
반드시 `setCount(count + 1)`을 호출해야 React가 화면을 다시 그린다.

**흔한 실수**:
```tsx
// ❌ 이렇게 하면 화면이 안 바뀜
count = count + 1;

// ✅ 이렇게 해야 화면이 바뀜
setCount(count + 1);
```

---

## 이해 확인

아래 코드에서 버튼을 클릭하면 어떤 일이 일어나는지 설명해볼 수 있나요?

```tsx
const [count, setCount] = useState(0);
<button onClick={() => setCount(count + 1)}>+1</button>
```

**흐름**: 버튼 클릭 → `onClick` 실행 → `setCount(count + 1)` 호출 → React가 새 값으로 화면 다시 그림 → 숫자가 1 증가한 화면이 보임
