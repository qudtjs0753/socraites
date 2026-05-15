'use client'; // 버튼 클릭 = 브라우저 이벤트 → 이 줄이 필요

import { useState } from 'react';

// 컴포넌트 = 화면의 부품을 만드는 함수
// 이 함수는 JSX(HTML처럼 생긴 것)를 반환한다
export default function CounterPage() {
  // useState: 화면에 영향을 주는 변수를 만드는 방법
  //   count    = 현재 값 (직접 수정 금지)
  //   setCount = 값을 바꾸는 함수 (이걸 호출해야 화면이 업데이트됨)
  //   useState(0) = 초기값은 0
  const [count, setCount] = useState(0);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-8">
      <h1 className="text-3xl font-bold">나의 첫 번째 React 컴포넌트</h1>

      {/* count 값이 0 미만이면 빨간색, 아니면 검은색 */}
      <p className={`text-6xl font-bold ${count < 0 ? 'text-red-500' : 'text-gray-900'}`}>
        {count}
      </p>

      <div className="flex gap-4">
        <button
          onClick={() => setCount(count - 1)} // 클릭 → setCount 호출 → 화면 업데이트
          className="rounded-lg bg-red-500 px-6 py-3 text-lg font-semibold text-white hover:bg-red-600"
        >
          -1
        </button>
        <button
          onClick={() => setCount(count + 1)}
          className="rounded-lg bg-blue-500 px-6 py-3 text-lg font-semibold text-white hover:bg-blue-600"
        >
          +1
        </button>
      </div>

      {/* 현재 상태에 대한 설명 */}
      <p className="text-gray-500">
        {count === 0 && '0이에요. 버튼을 눌러보세요!'}
        {count > 0 && `${count}만큼 올라갔어요 🎉`}
        {count < 0 && `${Math.abs(count)}만큼 내려갔어요 📉`}
      </p>
    </div>
  );
}
