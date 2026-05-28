# 훅(Hook) — 왜 만들어졌는가

## 훅이란

`use`로 시작하는 함수. React가 제공하는 기능(상태, 생명주기 등)을 **함수 컴포넌트 안에서 꺼내 쓸 수 있게 해주는 함수**다.

```tsx
const [count, setCount] = useState(0); // 상태 관리
useEffect(() => { fetchData(); }, []);  // 렌더링 후 실행
```

훅은 특별한 문법이 아니다. 그냥 함수다. `useState(0)`은 `[현재값, 변경함수]` 배열을 리턴하는 일반 함수 호출이다.

---

## 훅이 생기기 전 — 클래스 컴포넌트의 세 가지 문제

React 공식 문서([Hooks Intro - Motivation](https://legacy.reactjs.org/docs/hooks-intro.html#motivation))에 명시된 세 가지 이유다.

---

### 문제 1. 컴포넌트 간 상태 로직 재사용이 어렵다 ← 가장 큰 이유

"창 크기 추적" 로직을 여러 컴포넌트에서 쓰고 싶다면? 클래스 컴포넌트에서는 **로직이 항상 컴포넌트 안에 갇혀있었다.** 로직만 뽑아낼 방법이 없었다.

그래서 "로직을 가진 컴포넌트로 감싸는" HOC(Higher Order Component) 패턴이 등장했다:

```jsx
function withWindowSize(WrappedComponent) {
  return class extends React.Component {
    state = { width: window.innerWidth };
    componentDidMount() {
      window.addEventListener('resize', () =>
        this.setState({ width: window.innerWidth })
      );
    }
    render() {
      return <WrappedComponent windowWidth={this.state.width} {...this.props} />;
    }
  };
}

const ProfilePage = withWindowSize(ProfilePageBase);
const SideBar     = withWindowSize(SideBarBase);
```

이게 쌓이면 래퍼가 겹겹이 쌓인 "래퍼 지옥"이 됐다:

```jsx
export default withAuth(
  withWindowSize(
    withTheme(
      withRouter(MyComponent)
    )
  )
);
```

`windowWidth`가 어디서 오는지 추적하려면 파일 여러 개를 열어야 했다.

**훅으로 해결**: 컴포넌트 구조를 건드리지 않고 로직만 분리한다.

```jsx
function useWindowSize() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handler = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, []);
  return width;
}

function ProfilePage() { const width = useWindowSize(); } // 한 줄 재사용
function SideBar()     { const width = useWindowSize(); }
```

---

### 문제 2. 복잡한 컴포넌트는 이해하기 어려워진다

두 가지 방향으로 코드가 엉켰다.

**방향 A — 관련 코드가 흩어진다**: 채팅 구독 시작은 `componentDidMount`, 해제는 `componentWillUnmount`, 재구독은 `componentDidUpdate`에 각각 나뉜다.

**방향 B — 관계없는 코드가 뭉친다**: "마운트됐을 때"라는 이유만으로 유저 정보, 채팅, 분석이 한 메서드 안에 강제로 섞인다.

```jsx
componentDidMount() {
  fetchUser(this.props.userId)...  // 유저 정보
  this.chatSub = subscribeChatRoom(this.props.roomId); // 채팅
  analytics.track('page_view');   // 분석
}
componentWillUnmount() {
  this.chatSub.unsubscribe(); // 채팅 해제가 여기 따로 있음
}
```

**훅으로 해결**: 생명주기 이름이 아닌 **관심사** 기준으로 코드를 묶는다.

```jsx
// 유저 정보 — 관련 코드가 한 블록에
useEffect(() => {
  fetchUser(userId).then(setUser);
}, [userId]);

// 채팅 — 구독·재구독·해제가 한 블록에
useEffect(() => {
  const sub = subscribeChatRoom(roomId);
  return () => sub.unsubscribe(); // 시작과 정리가 붙어있음
}, [roomId]);
```

---

### 문제 3. 클래스는 사람도 기계도 혼란스럽게 한다

클래스 컴포넌트에서 가장 흔한 버그:

```jsx
class Button extends React.Component {
  handleClick() {
    console.log(this.state.count); // 오류: this가 undefined
  }
  render() {
    return <button onClick={this.handleClick}>클릭</button>;
    // 이벤트 핸들러로 전달하는 순간 this가 끊어진다
  }
}
```

해결하려면 `bind(this)` 또는 화살표 함수 필드를 써야 했다. JS를 처음 배우는 사람에게 `this`는 이미 어렵다. 클래스까지 더하면 진입 장벽이 높아진다.

⚠️ **기계(번들러) 문제**: 클래스는 어떤 메서드가 실제로 쓰이는지 정적으로 분석하기 어렵다. 안 쓰는 코드를 제거하는 **트리 쉐이킹(tree shaking)** 이 잘 안 돼서 번들 크기가 커진다.

**훅으로 해결**: 함수 컴포넌트에는 `this`가 없다.

```jsx
function Button() {
  const [count, setCount] = useState(0);
  const handleClick = () => {
    console.log(count); // this 없이 그냥 동작
  };
  return <button onClick={handleClick}>{count}</button>;
}
```

---

## 세 문제 요약

| 문제 | 원인 | 훅의 해결 |
|------|------|---------|
| 로직 재사용 불가 | 로직이 컴포넌트 안에 갇힘 | 커스텀 훅으로 로직만 분리 |
| 관련 코드가 흩어짐 | 생명주기 메서드 기준으로 묶임 | useEffect로 관심사 기준으로 묶음 |
| `this` 혼란 | 클래스 구조 자체의 문제 | 함수 컴포넌트에는 `this` 없음 |

세 문제 중 **로직 재사용**이 공식 문서에서 가장 먼저, 가장 비중있게 다뤄진다.

---

## ⚠️ 훅의 두 가지 규칙

React는 훅을 **호출 순서**로 추적한다. 순서가 바뀌면 어떤 훅이 어떤 상태인지 추적을 잃는다.

**규칙 1: 컴포넌트(또는 커스텀 훅) 최상단에서만 호출**

```tsx
// ❌ 조건문 안에서 호출 금지
if (someCondition) {
  const [value, setValue] = useState(0);
}

// ✅ 항상 최상단에서
const [value, setValue] = useState(0);
if (someCondition) { /* 사용은 여기서 */ }
```

**규칙 2: 일반 JS 함수 안에서 호출 금지**

```tsx
// ❌ 일반 함수 안 금지
function normalFunction() {
  const [v, setV] = useState(0);
}

// ✅ use로 시작하는 함수(커스텀 훅) 또는 컴포넌트 안에서만
function useMyHook() {
  const [v, setV] = useState(0);
}
```

---

## Best Case / Worst Case

### Best Case — 훅이 빛을 발하는 상황

- **로직이 여러 컴포넌트에서 반복될 때**: 커스텀 훅으로 분리하면 한 줄 재사용
- **시작과 정리가 항상 쌍으로 다닐 때**: `useEffect`의 cleanup 함수로 한 블록에 표현
- **관심사가 명확히 나뉠 때**: `useEffect`를 여러 개로 분리해서 각자 독립적으로 관리

### Worst Case — 훅이 오히려 복잡해지는 상황

- **훅 안에서 또 훅을 남발**: 커스텀 훅이 너무 많은 역할을 하면 오히려 추적이 어려워진다
- **의존성 배열을 잘못 관리**: `useEffect`에 필요한 변수를 빠뜨리면 오래된 값(stale closure)을 참조하는 버그가 생긴다
- **`use` prefix 없는 커스텀 훅**: React가 훅으로 인식하지 않아 규칙 검사를 건너뛰고 버그가 숨는다
