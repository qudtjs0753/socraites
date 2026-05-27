# Stage 2 — useEffect 심화: StopwatchApp

스톱워치와 창 너비 표시기를 만들면서 `useEffect`의 cleanup, `setInterval`, `addEventListener` 패턴을 익힌다.

## 파일 위치

파일은 프로젝트에 자동으로 복사되었습니다.

`_workspace/my-first-app/app/stage2/page.tsx`

수동으로 복사하려면:

```bash
cp -r _workspace/examples/stage2-useeffect/* my-first-app/app/stage2/
```

원본 파일명: `_workspace/examples/stage2-useeffect/page.tsx`

## 실행

```bash
cd _workspace/my-first-app
npm run dev
# http://localhost:3000/stage2
```

## 개념 참고

→ **[concepts.md](./concepts.md)** — useEffect 기본 원리, 의존성 배열, cleanup, 흔한 실수 정리

---

## 이 예제에서 다루는 패턴

### 1. cleanup 함수 — 타이머 정리

`useEffect`가 반환하는 함수는
**다음 effect가 실행되기 직전** 또는 **컴포넌트가 사라질 때** 호출된다.
정리하지 않으면 setInterval이 계속 살아남아 메모리 누수와 중복 실행이 발생한다.

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

### 2. setInterval + useEffect 연동

interval은 외부 시스템(브라우저 타이머)이라 React 상태 흐름 바깥에 있다.
`useEffect`로 감싸야 컴포넌트의 생명주기와 동기화할 수 있다.

```tsx
const id = setInterval(() => {
  setElapsed((prev) => prev + 10); // ← 함수형 업데이트: 최신 prev 보장
}, 10);
```

흔한 실수: `setElapsed(elapsed + 10)`로 쓰면 effect가 처음 만든 `elapsed` 값(클로저)을 영원히 참조하게 된다. 함수형 업데이트를 써야 한다.

### 3. window 이벤트 리스너 + useEffect 연동

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

### 4. 의존성 배열 패턴

- `[running]`: running이 바뀔 때마다 effect 재실행 → 새 interval 시작 (이전 건 cleanup으로 정리됨)
- `[]`: 1회만 등록 (resize 리스너처럼 상태와 무관한 외부 이벤트)
- 의존성 누락 시 ESLint(`react-hooks/exhaustive-deps`)가 경고

## 컴포넌트 구조

```
StopwatchApp
├─ useState: elapsed, running, width
├─ useEffect([running])  → setInterval / clearInterval
├─ useEffect([])         → addEventListener / removeEventListener
└─ <main>
    ├─ Stopwatch section (시간 + Start/Stop/Reset)
    └─ Window Width section
```

## 체크포인트

- [ ] Start 누르면 시간이 0.01초 단위로 증가
- [ ] Stop 누르면 멈춤, 다시 Start하면 이어서 진행
- [ ] Reset 누르면 0으로 초기화되고 정지
- [ ] 브라우저 창 크기를 바꾸면 너비 표시값이 실시간 갱신
- [ ] React DevTools/콘솔에서 경고가 없는지 확인

## 확장 과제

### 쉬움
- 분/초/밀리초 형식(`mm:ss.ms`)으로 표시하기

### 보통
- Lap(랩) 기능 추가: Start 중에 "Lap" 버튼을 누르면 현재 시간이 리스트에 쌓이도록

### 도전
- 의존성 배열을 일부러 잘못 써보기: `setElapsed(elapsed + 10)`로 바꾸고 동작이 어떻게 깨지는지 확인 → 왜 그런지 cleanup/클로저 관점에서 설명해보기

## Q&A 기록

<!-- 확장 과제 진행 중 질문과 해결책을 여기 기록하세요.
형식:
### Q. (질문)
**시도한 해결책 1**
```tsx
// 코드
```
- 장점:
- 단점:
-->
