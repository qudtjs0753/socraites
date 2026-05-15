# Stage 1 실습: Counter 앱

## 배우는 것

- 컴포넌트: 화면의 부품을 함수로 만드는 방법
- JSX: JS 안에서 HTML처럼 쓰는 문법
- useState: 화면을 바꾸는 변수
- 이벤트 처리: 버튼 클릭에 반응하기

## 실행 방법

```bash
# 1. Next.js 프로젝트 생성 (처음 한 번만)
npx create-next-app@latest my-first-app --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"

# 2. 프로젝트 폴더로 이동
cd my-first-app

# 3. 개발 서버 시작
npm run dev

# 4. 브라우저에서 열기: http://localhost:3000/counter
```

## 파일 위치

이 폴더의 `page.tsx` 파일을 `my-first-app/src/app/counter/page.tsx` 에 복사하세요.

```bash
# 터미널에서 한 번에 복사하는 방법 (경로는 실제 위치에 맞게 조정)
cp page.tsx my-first-app/src/app/counter/page.tsx
```

## 체크포인트 ✅

- [ ] `npm run dev` 실행 후 http://localhost:3000/counter 접속 가능
- [ ] +1 버튼 클릭 시 숫자 증가
- [ ] -1 버튼 클릭 시 숫자 감소
- [ ] 0 미만이면 숫자가 빨간색으로 표시

## 확장 과제 🔧

1. **쉬움**: 리셋 버튼 추가 (클릭하면 0으로 돌아가기)
2. **보통**: 최솟값(-10)과 최댓값(10) 제한 추가
3. **도전**: 카운터를 2개 만들고, 두 카운터의 합을 아래에 표시
