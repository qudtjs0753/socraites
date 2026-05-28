'use client';

import { useCallback, useEffect, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

// ─────────────────────────────────────────────
// Custom Hook 1: useLocalStorage
// useState처럼 동작하지만 값이 바뀔 때마다 localStorage에 자동 저장
// ─────────────────────────────────────────────
function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(initialValue);

  // mount 시 localStorage에서 값 읽기 (SSR 대비: useEffect 안에서)
  useEffect(() => {
    const raw = window.localStorage.getItem(key);
    if (raw !== null) {
      try {
        setValue(JSON.parse(raw) as T);
      } catch {
        // 파싱 실패 시 초기값 유지
      }
    }
  }, [key]);

  // value가 바뀔 때마다 localStorage에 저장
  useEffect(() => {
    window.localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
}

// ─────────────────────────────────────────────
// Custom Hook 2: useToggle
// boolean 상태와 토글 함수를 반환
// ─────────────────────────────────────────────
function useToggle(initial = false): [boolean, () => void] {
  const [on, setOn] = useState(initial);
  const toggle = useCallback(() => setOn((prev) => !prev), []);
  return [on, toggle];
}

// ─────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────
type Memo = {
  id: string;
  text: string;
  done: boolean;
};

export default function LocalMemoApp() {
  const [memos, setMemos] = useLocalStorage<Memo[]>('local-memos', []);
  const [hideDone, toggleHideDone] = useToggle(false);
  const [draft, setDraft] = useState('');

  const addMemo = () => {
    const text = draft.trim();
    if (!text) return;
    const newMemo: Memo = {
      id: crypto.randomUUID(),
      text,
      done: false,
    };
    setMemos((prev) => [newMemo, ...prev]);
    setDraft('');
  };

  const toggleDone = (id: string) => {
    setMemos((prev) =>
      prev.map((m) => (m.id === id ? { ...m, done: !m.done } : m)),
    );
  };

  const removeMemo = (id: string) => {
    setMemos((prev) => prev.filter((m) => m.id !== id));
  };

  const visible = hideDone ? memos.filter((m) => !m.done) : memos;

  return (
    <main className="mx-auto max-w-md p-8 space-y-6">
      <header>
        <h1 className="text-2xl font-bold">LocalMemo</h1>
        <p className="text-sm text-zinc-500">
          새로고침해도 사라지지 않는 메모장
        </p>
      </header>

      <section className="flex gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && addMemo()}
          placeholder="메모를 입력하세요"
          className="flex-1 rounded border border-zinc-300 px-3 py-2"
        />
        <button
          onClick={addMemo}
          className="rounded bg-blue-500 px-4 py-2 text-white"
        >
          추가
        </button>
      </section>

      <section className="flex items-center justify-between">
        <span className="text-sm text-zinc-500">
          총 {memos.length}개 / 완료 {memos.filter((m) => m.done).length}개
        </span>
        <button
          onClick={toggleHideDone}
          className="rounded border border-zinc-300 px-3 py-1 text-sm"
        >
          {hideDone ? '완료 보이기' : '완료 숨기기'}
        </button>
      </section>

      <ul className="space-y-2">
        {visible.map((memo) => (
          <li
            key={memo.id}
            className="flex items-center gap-3 rounded border border-zinc-200 px-3 py-2"
          >
            <input
              type="checkbox"
              checked={memo.done}
              onChange={() => toggleDone(memo.id)}
            />
            <span
              className={`flex-1 ${memo.done ? 'text-zinc-400 line-through' : ''}`}
            >
              {memo.text}
            </span>
            <button
              onClick={() => removeMemo(memo.id)}
              className="text-sm text-red-500"
            >
              삭제
            </button>
          </li>
        ))}
        {visible.length === 0 && (
          <li className="text-center text-sm text-zinc-400">
            메모가 없습니다
          </li>
        )}
      </ul>
    </main>
  );
}
