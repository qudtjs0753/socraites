'use client';

import { useState } from 'react';

// 할 일 하나의 데이터 구조 (TypeScript 타입)
interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

// ─────────────────────────────────────────────
// TodoItem 컴포넌트: 할 일 하나를 표시하는 부품
// props로 데이터(todo)와 동작(onToggle, onDelete)을 받는다
// ─────────────────────────────────────────────
function TodoItem({
  todo,
  onToggle,
  onDelete,
}: {
  todo: Todo;
  onToggle: (id: number) => void; // 완료 토글 함수 (부모에서 전달)
  onDelete: (id: number) => void; // 삭제 함수 (부모에서 전달)
}) {
  return (
    <li className="flex items-center gap-3 rounded-lg border bg-white p-3 shadow-sm">
      {/* 체크박스 클릭 → 부모의 onToggle 호출 */}
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => onToggle(todo.id)}
        className="h-4 w-4 cursor-pointer"
      />

      {/* 조건부 렌더링: completed면 취소선 + 회색 */}
      <span
        className={`flex-1 ${todo.completed ? 'text-gray-400 line-through' : 'text-gray-800'}`}
      >
        {todo.text}
      </span>

      {/* 삭제 버튼 클릭 → 부모의 onDelete 호출 */}
      <button
        onClick={() => onDelete(todo.id)}
        className="text-sm text-red-400 hover:text-red-600"
      >
        삭제
      </button>
    </li>
  );
}

// ─────────────────────────────────────────────
// TodoPage: 전체 페이지 컴포넌트 (상태 관리 담당)
// ─────────────────────────────────────────────
export default function TodoPage() {
  const [todos, setTodos] = useState<Todo[]>([
    { id: 1, text: '리액트 공부하기', completed: false },
    { id: 2, text: '예제 실행해보기', completed: true },
  ]);
  const [inputValue, setInputValue] = useState('');

  // 할 일 추가
  function handleAdd() {
    const trimmed = inputValue.trim();
    if (!trimmed) return; // 빈 문자열 무시

    const newTodo: Todo = {
      id: Date.now(), // 임시 고유 ID
      text: trimmed,
      completed: false,
    };
    setTodos([...todos, newTodo]); // 기존 배열에 새 항목 추가
    setInputValue(''); // 입력창 비우기
  }

  // Enter 키로도 추가
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') handleAdd();
  }

  // 완료 토글
  function handleToggle(id: number) {
    setTodos(
      todos.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo,
      ),
    );
  }

  // 삭제
  function handleDelete(id: number) {
    setTodos(todos.filter(todo => todo.id !== id));
  }

  const completedCount = todos.filter(t => t.completed).length;

  return (
    <div className="mx-auto max-w-md px-4 py-12">
      <h1 className="mb-2 text-2xl font-bold text-gray-900">할 일 목록</h1>

      {/* 진행 현황 */}
      <p className="mb-6 text-sm text-gray-500">
        전체 {todos.length}개 중 {completedCount}개 완료
      </p>

      {/* 입력 폼 */}
      <div className="mb-6 flex gap-2">
        <input
          type="text"
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="할 일을 입력하세요"
          className="flex-1 rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={handleAdd}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          추가
        </button>
      </div>

      {/* 조건부 렌더링: 빈 상태 */}
      {todos.length === 0 && (
        <p className="py-8 text-center text-gray-400">
          할 일이 없어요. 새로운 할 일을 추가해보세요! 🎉
        </p>
      )}

      {/* 리스트 렌더링: todos 배열 → TodoItem 컴포넌트 반복 */}
      <ul className="space-y-2">
        {todos.map(todo => (
          <TodoItem
            key={todo.id}           // 리스트 렌더링 필수: 고유 key
            todo={todo}             // props: 데이터 전달
            onToggle={handleToggle} // props: 함수 전달 (자식 → 부모 통신)
            onDelete={handleDelete} // props: 함수 전달
          />
        ))}
      </ul>
    </div>
  );
}
