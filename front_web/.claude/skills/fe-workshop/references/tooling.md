# 개발 도구 설정 레퍼런스

## TypeScript 기본 타입

### 자주 쓰는 타입 패턴

```tsx
// 문자열/숫자/불리언
const name: string = '홍길동';
const age: number = 25;
const isLoggedIn: boolean = true;

// 배열
const items: string[] = ['사과', '바나나'];
const ids: number[] = [1, 2, 3];

// 객체 타입 (interface)
interface User {
  id: number;
  name: string;
  email: string;
  createdAt?: Date; // 선택 사항 (? = 없어도 됨)
}

// 유니언 타입 (이것 또는 저것)
type Status = 'pending' | 'active' | 'inactive';
type StringOrNumber = string | number;

// 제네릭 (다양한 타입에 재사용)
interface ApiResponse<T> {
  data: T;
  error: string | null;
  status: number;
}
// 사용: ApiResponse<User[]>, ApiResponse<Todo>
```

### React + TypeScript 패턴

```tsx
// 컴포넌트 Props
interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary';
}

// 이벤트 타입
const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  setValue(e.target.value);
};

const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
  e.preventDefault();
};

// useState 타입 추론 (자동으로 됨)
const [count, setCount] = useState(0);         // number로 추론
const [name, setName] = useState('');          // string으로 추론
const [user, setUser] = useState<User | null>(null); // 명시 필요
```

## VS Code 설정

### 추천 확장 (Extensions)

```
필수:
- ESLint (Microsoft)
- Prettier - Code formatter
- Tailwind CSS IntelliSense
- TypeScript (내장)

편의:
- Auto Import
- Path Intellisense
- GitLens
```

### .vscode/settings.json

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  },
  "typescript.preferences.importModuleSpecifier": "non-relative",
  "tailwindCSS.experimental.classRegex": [
    ["cn\\(([^)]*)\\)", "(?:'|\"|`)([^']*)(?:'|\"|`)"]
  ]
}
```

## next.config.ts 기본 설정

```ts
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // 외부 이미지 도메인 허용
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
      },
    ],
  },
  // 환경 변수 타입 검사 (선택)
  experimental: {
    typedRoutes: true,
  },
};

export default nextConfig;
```

## Git 커밋 컨벤션 (Conventional Commits)

```
형식: <타입>: <제목>

타입:
  feat     - 새 기능
  fix      - 버그 수정
  docs     - 문서 변경
  style    - 포맷, 세미콜론 등 (기능 변경 없음)
  refactor - 리팩토링
  test     - 테스트 추가/수정
  chore    - 빌드, 의존성 등

예시:
  feat: 로그인 기능 추가
  fix: 버튼 클릭 시 이벤트 중복 발생 수정
  docs: README 설치 방법 업데이트
  test: Counter 컴포넌트 테스트 추가
```
