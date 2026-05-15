# 학습 단계별 상세 내용

## 목차

1. [Stage 0: JS 필수 개념 보강](#stage-0)
2. [Stage 1: React 기초](#stage-1)
3. [Stage 2: React 심화](#stage-2)
4. [Stage 3: Next.js 기초](#stage-3)
5. [Stage 4: TanStack Query](#stage-4)
6. [Stage 5: Tailwind CSS](#stage-5)
7. [Stage 6: 개발 도구](#stage-6)
8. [Stage 7: 테스트 코드](#stage-7)
9. [Stage 8: 배포](#stage-8)

---

## Stage 0: JS 필수 개념 보강 {#stage-0}

**대상**: JS 변수/함수는 알지만 최신 문법이 낯선 경우  
**예상 기간**: 3~5일

### 핵심 개념

**화살표 함수**
```js
// 기존 함수
function add(a, b) { return a + b; }

// 화살표 함수 (React에서 매우 자주 씀)
const add = (a, b) => a + b;
```

**구조 분해 할당**
```js
// 객체
const { name, age } = user;

// 배열
const [first, second] = ['사과', '바나나'];

// React props에서 자주 씀
function Card({ title, content }) { ... }
```

**스프레드 연산자**
```js
const newArray = [...oldArray, newItem];
const newObj = { ...oldObj, newKey: 'value' };
```

**템플릿 리터럴**
```js
const message = `안녕하세요, ${name}님!`;
```

**배열 메서드**
```js
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
```

**async/await**
```js
async function loadData() {
  const response = await fetch('/api/data');
  const data = await response.json();
  return data;
}
```

**모듈 시스템**
```js
// 내보내기
export function MyComponent() { ... }
export default function App() { ... }

// 가져오기
import { MyComponent } from './MyComponent';
import App from './App';
```

### 완료 기준
- 위 문법들을 보고 이해할 수 있다 (암기 불필요)
- MDN 또는 검색으로 찾아볼 수 있다

---

## Stage 1: React 기초 {#stage-1}

**예상 기간**: 1~2주

### 핵심 개념

1. **컴포넌트와 JSX**: 화면의 부품을 함수로 만드는 방법
2. **Props**: 부모 컴포넌트에서 자식에게 데이터 전달
3. **State (useState)**: 화면을 업데이트하는 변수
4. **이벤트 처리**: 클릭, 입력 등 사용자 행동에 반응
5. **조건부 렌더링**: 상황에 따라 다른 UI 표시
6. **리스트 렌더링**: 배열 데이터를 화면에 나열
7. **useEffect 기초**: 컴포넌트 렌더링 후 실행할 코드

### 실습 과제

```
과제 1: Counter 앱
- +1, -1 버튼으로 숫자 변경
- 0 미만으로 내려가면 빨간 숫자 표시

과제 2: Todo 앱 (로컬 state만)
- 할 일 추가, 완료 체크, 삭제
- 완료된 항목 개수 표시

과제 3: 이미지 갤러리
- 사진 목록을 카드로 표시
- 클릭 시 모달로 확대
```

### 완료 기준
- useState로 인터랙티브한 컴포넌트를 스스로 만들 수 있다
- props를 통해 부모→자식 데이터 전달을 이해한다
- 리스트에서 `key` prop이 왜 필요한지 설명할 수 있다

---

## Stage 2: React 심화 {#stage-2}

**예상 기간**: 1주

### 핵심 개념

1. **Custom Hooks**: useState/useEffect 로직을 재사용하는 방법
2. **useContext**: props drilling 없이 전역 상태 공유
3. **useCallback + useMemo**: 불필요한 렌더링 방지 (성능 최적화)
4. **폼 처리 패턴**: controlled vs uncontrolled 입력

### 실습 과제

```
과제 1: useFetch 훅 만들기
- API 호출 로직을 훅으로 추출
- loading, error, data 상태 관리

과제 2: 테마 토글 (다크모드)
- Context로 테마 상태 공유
- useTheme 커스텀 훅

과제 3: 검색 + 필터
- 검색어 입력 → 디바운스 → 결과 표시
- useCallback으로 함수 최적화
```

### 완료 기준
- Custom Hook을 직접 만들어서 분리할 수 있다
- Context가 언제 유용한지, 언제 오히려 복잡한지 설명할 수 있다

---

## Stage 3: Next.js 기초 {#stage-3}

**예상 기간**: 1~2주

### 핵심 개념

1. **App Router**: 폴더 = URL 경로
2. **Server Component**: 서버에서 렌더링, DB/API 직접 접근
3. **Client Component**: 브라우저에서 실행, `'use client'` 선언
4. **Layouts**: 여러 페이지에 공통 UI
5. **데이터 페칭 전략**:
   - SSR (Server-Side Rendering): 요청마다 최신 데이터
   - SSG (Static Site Generation): 빌드 시 1회 생성
   - ISR (Incremental Static Regeneration): 주기적으로 갱신
6. **Loading/Error UI**: `loading.tsx`, `error.tsx`

### 언제 무엇을 쓰는가

| 상황 | 선택 | 이유 |
|------|------|------|
| 사용자마다 다른 데이터 | SSR | 개인화 필요 |
| 모든 사용자 같은 정적 콘텐츠 | SSG | 빠른 속도 |
| 자주 바뀌는 공통 데이터 | ISR | 속도 + 최신성 균형 |
| 클릭/입력 인터랙션 | Client Component | 브라우저 이벤트 필요 |

### 실습 과제

```
과제 1: 블로그 목록 + 상세 페이지
- /blog → 목록 페이지
- /blog/[slug] → 상세 페이지
- SSG + ISR 적용

과제 2: 인증 필요 페이지
- 로그인 여부에 따라 다른 UI
- middleware로 미인증 redirect
```

### 완료 기준
- Server Component와 Client Component를 올바르게 구분할 수 있다
- 폴더 구조만 보고 URL 경로를 예측할 수 있다

---

## Stage 4: TanStack Query {#stage-4}

**예상 기간**: 1주

### 핵심 개념

1. **서버 상태 vs 클라이언트 상태**: 왜 구분하는가
2. **useQuery**: 데이터 가져오기
3. **useMutation**: 데이터 변경 (생성/수정/삭제)
4. **캐싱과 무효화**: `queryKey`, `invalidateQueries`
5. **낙관적 업데이트**: 서버 응답 전에 UI 먼저 업데이트

### 실습 과제

```
과제 1: 게시판
- 목록 조회 (useQuery)
- 글 작성 (useMutation + invalidateQueries)
- 글 삭제 (낙관적 업데이트)

과제 2: 무한 스크롤
- useInfiniteQuery
- Intersection Observer
```

### 완료 기준
- useQuery를 useEffect + useState 대신 쓰는 이유를 설명할 수 있다
- mutation 후 목록이 자동 갱신되게 만들 수 있다

---

## Stage 5: Tailwind CSS {#stage-5}

**예상 기간**: 3~4일

### 핵심 개념

1. **Utility-First**: 클래스명 하나가 CSS 속성 하나
2. **반응형**: `sm:` `md:` `lg:` 접두사
3. **상태**: `hover:` `focus:` `disabled:` 접두사
4. **다크 모드**: `dark:` 접두사
5. **커스터마이징**: `tailwind.config.ts`

### 실습 과제

```
과제 1: 카드 컴포넌트
- 반응형 그리드 레이아웃
- 호버 애니메이션

과제 2: 네비게이션 바
- 데스크탑: 가로 메뉴
- 모바일: 햄버거 메뉴

과제 3: 폼 UI
- Input, Button, Label 컴포넌트화
- 포커스/에러 상태 스타일
```

### 완료 기준
- 디자인 figma/이미지를 보고 Tailwind로 구현할 수 있다
- 반응형 레이아웃을 만들 수 있다

---

## Stage 6: 개발 도구 {#stage-6}

**예상 기간**: 2~3일

### 항목

- TypeScript 기본 타입 (props, API 응답, 이벤트)
- ESLint 규칙 설정 + 에러 수정
- Prettier 자동 포맷팅
- VS Code 확장 (ESLint, Prettier, Tailwind IntelliSense)
- Git 커밋 컨벤션 (Conventional Commits)

### 완료 기준
- 저장 시 자동 포맷팅이 적용된다
- TypeScript 타입 오류가 IDE에서 바로 보인다

---

## Stage 7: 테스트 코드 {#stage-7}

**예상 기간**: 1주

### 항목

- Vitest + Testing Library 환경 설정
- 컴포넌트 렌더링 테스트
- 사용자 인터랙션 테스트 (userEvent)
- useQuery 컴포넌트 테스트 (MSW 모킹)
- Custom Hook 테스트

### 완료 기준
- 작성한 컴포넌트에 테스트 2~3개를 스스로 작성할 수 있다
- 테스트가 실패하면 무엇이 잘못됐는지 읽을 수 있다

---

## Stage 8: 배포 {#stage-8}

**예상 기간**: 2~3일

### 항목

- `npm run build` 로컬 빌드 확인
- Vercel 프로젝트 생성 + 배포
- GitHub 연동 자동 배포
- 환경 변수 설정
- GitHub Actions CI/CD (빌드 + 테스트 자동화)

### 완료 기준
- 코드 push 후 자동으로 배포된다
- 실제 URL로 프로젝트에 접근할 수 있다
