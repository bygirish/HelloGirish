"use client";


import React, { useState, useEffect } from 'react';
import "./styles.css";  // Assuming you're using SCSS for styling

const AnalogClock = () => {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const intervalId = setInterval(() => {
      setTime(new Date());
    }, 1000);
    return () => clearInterval(intervalId);
  }, []);

  // Get current hours, minutes, and seconds
  const seconds = time.getSeconds();
  const minutes = time.getMinutes();
  const hours = time.getHours();

  console.log("hours: " + hours, "minutes: " + minutes, "seconds: " + seconds);

  const baseDeg = 270;
  // Calculate rotation angles for clock hands
  const secondsDeg = baseDeg + (seconds / 60) * 360;
  const minutesDeg = baseDeg + (minutes / 60) * 360 + (seconds / 60) * 6;
  const hoursDeg = baseDeg + (hours / 12) * 360 + (minutes / 60) * 30;

  // const secondsDeg = baseDeg;
  // const minutesDeg = baseDeg;
  // const hoursDeg = baseDeg;

  console.log("hours: " + hoursDeg, "minutes: " + minutesDeg, "seconds: " + secondsDeg);

  return (
    <div className="clock">
      {/* <div className="clock-face"> */}
        <div
          className="hand hour-hand"
          style={{ transform: `rotate(${hoursDeg}deg)` }}
        ></div>
        <div
          className="hand minute-hand"
          style={{ transform: `rotate(${minutesDeg}deg)` }}
        ></div>
        <div
          className="hand second-hand"
          style={{ transform: `rotate(${secondsDeg}deg)` }}
        ></div>
      {/* </div> */}
    </div>
  );
};

export default AnalogClock;
