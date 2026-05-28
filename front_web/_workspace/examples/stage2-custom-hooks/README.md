# Stage 2 — Custom Hooks (LocalMemo)

`useState`/`useEffect`를 조합해 **나만의 훅**을 만들고, 그 훅을 컴포넌트에서 재사용하는 패턴을 익힌다.

## 무엇을 만드나

- **LocalMemo**: 새로고침해도 사라지지 않는 메모장
- 메모 추가 / 삭제 / 완료 토글 / 완료 숨기기

## 직접 만드는 훅 2개

| 훅 | 역할 |
|---|---|
| `useLocalStorage<T>` | `useState`처럼 동작하지만 값 변경 시 자동으로 `localStorage`에 저장 |
| `useToggle` | boolean 상태와 `toggle` 함수를 반환 |

## 실행 방법

```bash
cd _workspace/my-first-app
npm run dev
```

브라우저에서 `http://localhost:3000/stage2-custom-hooks` 접속.

## 파일 위치

파일은 프로젝트에 자동으로 복사되었습니다.

`_workspace/my-first-app/src/app/stage2-custom-hooks/page.tsx`

원본 파일은 `_workspace/examples/stage2-custom-hooks/page.tsx`이며, 수동으로 복사하려면:

```bash
cp _workspace/examples/stage2-custom-hooks/page.tsx \
   _workspace/my-first-app/src/app/stage2-custom-hooks/page.tsx
```

## 체크포인트 ✅

1. 메모를 추가하면 리스트 상단에 나타난다.
2. 체크박스를 누르면 취소선과 함께 회색으로 바뀐다.
3. **새로고침해도** 메모가 그대로 남아 있다. (← `useLocalStorage` 동작 확인)
4. "완료 숨기기" 버튼을 누르면 완료 항목이 사라진다. (← `useToggle` 동작 확인)
5. 브라우저 DevTools → Application → Local Storage에서 `local-memos` 키 확인.

## 확장 과제 🔧

### 쉬움
- 메모 개수가 0일 때 입력창 placeholder를 "첫 메모를 적어보세요"로 변경.

### 보통
- `useLocalStorage`에 **초기값을 함수로 전달**할 수 있도록 개선
  (`useState`처럼 `() => expensive()` 형태 지원).

### 도전
- `useDebouncedValue<T>(value, ms)` 훅을 추가로 만들어,
  입력창에 글자를 칠 때마다 즉시 저장되지 않고 **0.5초 후에만** 저장되도록 변경.
  (현재는 매 키 입력마다 localStorage에 쓰여서 비효율적)

## Q&A 기록

> 확장 과제를 풀거나 코드를 읽다가 생긴 질문과 해결 과정을 여기에 남깁니다.

### Q1.

**질문:**

**해결 방법:**

**장단점:**

---

### Q2.

**질문:**

**해결 방법:**

**장단점:**
