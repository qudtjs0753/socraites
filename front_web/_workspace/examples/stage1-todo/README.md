# Stage 1 실습: Todo 앱

## 배우는 것

- **props**: 컴포넌트 간 데이터/함수 전달
- **리스트 렌더링**: `map` + `key`
- **조건부 렌더링**: `&&`, 삼항 연산자

## 파일 위치

이 폴더의 `page.tsx` 파일을 기존 프로젝트의 `src/app/todo/page.tsx`에 복사하세요.

```bash
mkdir -p my-first-app/src/app/todo
cp page.tsx my-first-app/src/app/todo/page.tsx
```

## 실행 방법

```bash
cd my-first-app
npm run dev
# http://localhost:3000/todo 접속
```

## 체크포인트 ✅

- [ ] 입력창에 텍스트 입력 후 "추가" 버튼 클릭 → 목록에 추가됨
- [ ] Enter 키로도 추가됨
- [ ] 체크박스 클릭 → 취소선 + 회색으로 변함
- [ ] 삭제 버튼 클릭 → 항목 제거됨
- [ ] 모든 항목 삭제 시 "할 일이 없어요" 메시지 표시

## 확장 과제 🔧

1. **쉬움**: "완료된 항목 모두 삭제" 버튼 추가
2. **보통**: 미완료 항목만 / 완료 항목만 / 전체 보기 필터 추가
3. **도전**: `localStorage`에 저장해서 새로고침해도 유지되게 만들기
   - 힌트: `JSON.stringify` / `JSON.parse` + `useEffect`

---

## Q&A 기록

### Q. 필터 기능, CSS로 숨기면 안 되나?

"미완료만 보기" 필터를 구현할 때 CSS `display: none`으로 숨기는 방법과 JS `.filter()`로 걸러서 렌더링하는 방법 중 어느 게 맞나?

**방법 1: CSS 방식**
```tsx
<li style={{ display: todo.completed ? 'none' : 'block' }}>
```
- 장점: 간단하게 구현 가능
- 단점: 요소가 DOM에 계속 존재함. React의 사고방식(데이터 → 화면)과 맞지 않음

**방법 2: React 방식 (권장)**
```tsx
const visibleTodos = showOnlyIncomplete
  ? todos.filter(t => !t.completed)
  : todos;
```
- 장점: 원본 `todos`는 건드리지 않고, 렌더링용 "뷰"만 따로 계산. 데이터와 화면이 명확히 분리됨
- 단점: 렌더링마다 새 배열 생성 (수만 개 수준이 아니면 문제 없음)

### Q. 토글할 때마다 새 배열을 만드는 게 맞나?

React는 상태가 바뀌면 컴포넌트 함수 전체를 다시 실행하는 구조라서, `visibleTodos` 계산도 매 렌더링마다 새로 함. 이게 정상이고 일반적인 방식.

수만 개 수준의 대용량 리스트라면 `useMemo`로 캐싱:
```tsx
const visibleTodos = useMemo(
  () => showOnlyIncomplete ? todos.filter(t => !t.completed) : todos,
  [todos, showOnlyIncomplete] // 이 두 값이 바뀔 때만 재계산
);
```
투두 앱 수준에서는 `useMemo` 없이 쓰는 게 보통. 조기 최적화는 코드만 복잡하게 만든다.

### Q. "React의 사고방식"이 뭔가? CSS 방식이 왜 맞지 않는다는 건가?

바닐라 JS는 화면 요소를 직접 찾아서 조작한다. React는 이걸 뒤집었다.

> **"화면을 직접 바꾸지 마라. 데이터만 바꿔라. 화면은 React가 알아서 그린다."**

```tsx
// ❌ 바닐라 JS 방식 — 화면을 직접 조작
document.getElementById('todo-1').style.display = 'none';

// ❌ CSS 방식 — 데이터엔 있는데 화면엔 없는 불일치 상태 발생
<li style={{ display: todo.completed ? 'none' : 'block' }}>

// ✅ React 방식 — 데이터를 먼저 가공, 화면은 그 결과만 표시
const visibleTodos = todos.filter(t => !t.completed);
```

CSS 방식이 "틀린" 건 아니지만 React의 약속("데이터가 항상 진실의 원천")을 깬다.
앱이 커질수록 화면과 데이터가 따로 노는 순간 디버깅이 어려워진다.

**실용적으로도 데이터 방식이 나은 이유:** 미완료 항목 개수를 표시하는 기능을 추가한다면,
CSS 방식은 숨긴 것도 세야 하는지 헷갈리지만 React 방식은 `todos.filter(t => !t.completed).length`로 끝난다.

### Q. useState와 useEffect의 실행 순서는?

`useState`가 먼저, `useEffect`는 화면이 그려진 다음에 실행된다.

```
① useState 초기값 설정 → ② 화면 렌더링 → ③ useEffect 실행
```

localStorage 연동 예시:

```tsx
const [todos, setTodos] = useState([]);  // ① [] 로 시작

useEffect(() => {
  const saved = localStorage.getItem('todos');
  setTodos(saved ? JSON.parse(saved) : []);  // ③ 화면 그려진 후 localStorage 읽어서 업데이트
}, []);
```

```
① useState([])    → todos = []
② 화면 렌더링     → 빈 목록 표시 (아주 잠깐)
③ useEffect 실행  → localStorage 읽어서 setTodos 호출
④ 다시 렌더링     → 저장된 목록 표시
```

`useEffect`는 의존성 배열 안의 값이 바뀔 때마다 실행된다.

```tsx
useEffect(() => { ... }, []);        // 최초 1회
useEffect(() => { ... }, [todos]);   // todos가 바뀔 때마다
useEffect(() => { ... });            // 렌더링마다 (배열 자체가 없음)
```
