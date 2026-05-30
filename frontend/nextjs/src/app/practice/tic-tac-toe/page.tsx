// https://dev.to/frontendengineer/reactjs-tic-tac-toe-passed-job-interview-1od8
// https://codepen.io/angelo_jin/pen/GROEdMK

"use client";

import React, { useState } from "react";
import "./styles.css";

type SquareEntryType = null | "X" | "O";

export default function Board() {
  const [squares, setSquares] = useState<SquareEntryType[]>(
    Array(9).fill(null)
  );
  const [isX, setIsX] = useState<boolean>(true);

  const [winner, setWinner] = useState<SquareEntryType>();

  const handleClick = (i: number) => {
    if (squares[i]) return;

    squares[i] = isX ? "X" : "O";
    setSquares(squares);

    const winnerFound = calculateWinner(squares);
    if (winnerFound) {
      setWinner(winnerFound);
      return;
    }

    setIsX(!isX);
  };

  const status = winner
    ? `Winner: ${winner}`
    : "Next player: " + (isX ? "X" : "O");

  const handleRestart = () => {
    setIsX(true);
    setSquares(Array(9).fill(null));
  };

  const renderSquare = (i: number) => {
    return <Square value={squares[i]} onClick={() => handleClick(i)} />;
  };

  return (
    <div className="board">
      <div className="board-row">
        {renderSquare(0)}
        {renderSquare(1)}
        {renderSquare(2)}
      </div>
      <div className="board-row">
        {renderSquare(3)}
        {renderSquare(4)}
        {renderSquare(5)}
      </div>
      <div className="board-row">
        {renderSquare(6)}
        {renderSquare(7)}
        {renderSquare(8)}
      </div>
      <div className="status">{status}</div>
      <button className="restart" onClick={handleRestart}>
        Restart Game!
      </button>
    </div>
  );
}

function Square({ onClick, value }: { onClick: any; value: SquareEntryType }) {
  return (
    <button className="square" onClick={onClick}>
      {value}
    </button>
  );
}

function calculateWinner(squares: any) {
  const winningPatterns = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];

  for (let i = 0; i < winningPatterns.length; i++) {
    const [a, b, c] = winningPatterns[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}
