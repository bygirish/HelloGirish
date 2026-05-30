// https://overreacted.io/making-setinterval-declarative-with-react-hooks/
"use client";

import { MutableRefObject, useEffect, useRef, useState } from "react";

export default function Counter() {
  const [count, setCount] = useState(0);

  useInterval(() => {
    setCount(count + 1);
  }, 1000);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <h1>Counter</h1>
      <h2>{count}</h2>
    </div>
  );
}

function useInterval(callback: Function, delay: number) {
  const savedCallback: MutableRefObject<Function | undefined> = useRef();

  useEffect(() => {
    savedCallback.current = callback;
  });

  useEffect(() => {
    function tick() {
      savedCallback?.current?.();
    }

    let id = setInterval(tick, delay);
    return () => clearInterval(id);
  }, [delay]);
}
