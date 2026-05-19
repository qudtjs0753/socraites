'use client';

import { useEffect, useState } from 'react';

export default function StopwatchApp() {
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const [width, setWidth] = useState(0);

  // 스톱워치: running이 true일 때만 interval 시작, cleanup으로 정리
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setElapsed((prev) => prev + 10);
    }, 10);
    return () => clearInterval(id); // cleanup
  }, [running]);

  // 창 너비: mount 시 1회 등록, unmount 시 제거
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    handleResize(); // 초기값
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize); // cleanup
  }, []);

  const seconds = (elapsed / 1000).toFixed(2);

  return (
    <main className="p-8 space-y-6">
      <section>
        <h1 className="text-2xl font-bold">Stopwatch</h1>
        <p className="text-4xl font-mono mt-2">{seconds}s</p>
        <div className="flex gap-2 mt-4">
          <button
            onClick={() => setRunning(true)}
            className="px-4 py-2 bg-green-500 text-white rounded"
          >
            Start
          </button>
          <button
            onClick={() => setRunning(false)}
            className="px-4 py-2 bg-yellow-500 text-white rounded"
          >
            Stop
          </button>
          <button
            onClick={() => {
              setRunning(false);
              setElapsed(0);
            }}
            className="px-4 py-2 bg-gray-500 text-white rounded"
          >
            Reset
          </button>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold">Window Width</h2>
        <p className="text-lg">{width}px</p>
      </section>
    </main>
  );
}
