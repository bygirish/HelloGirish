
// https://sonikamaheshwari067.medium.com/react-build-carousel-5dcd8848143b

"use client";

import "./styles.css";

import React, { useState } from "react";

const BsArrowLeftCircleFill = ({ onClick, className }) => {
  return (
    <div className={className} onClick={onClick}>
      {"Left"}
    </div>
  );
};

const BsArrowRightCircleFill = ({ onClick, className }) => {
  return (
    <div className={className} onClick={onClick}>
      {"Right"}
    </div>
  );
};

function App({}) {
  return (
    <div className="App">
      <Carousel data={slides} />
    </div>
  );
}

export default App;

const slides = [
  {
    src: "https://picsum.photos/seed/img1/600/400",
    alt: "Image 1 for carousel",
  },
  {
    src: "https://picsum.photos/seed/img2/600/400",
    alt: "Image 2 for carousel",
  },
  {
    src: "https://picsum.photos/seed/img3/600/400",
    alt: "Image 3 for carousel",
  },
];

export const Carousel = ({ data }) => {
  const [slide, setSlide] = useState(0);

  const nextSlide = () => {
    setSlide(slide === data.length - 1 ? 0 : slide + 1);
  };

  const prevSlide = () => {
    setSlide(slide === 0 ? data.length - 1 : slide - 1);
  };

  return (
    <div className="carousel">
      <BsArrowLeftCircleFill onClick={prevSlide} className="arrow arrow-left" />
      {data.map((item, idx) => {
        return (
          <img
            src={item.src}
            alt={item.alt}
            key={idx}
            className={slide === idx ? "slide" : "slide slide-hidden"}
          />
        );
      })}
      <BsArrowRightCircleFill
        onClick={nextSlide}
        className="arrow arrow-right"
      />
      <span className="indicators">
        {data.map((_, idx) => {
          return (
            <button
              key={idx}
              className={
                slide === idx ? "indicator" : "indicator indicator-inactive"
              }
              onClick={() => setSlide(idx)}
            ></button>
          );
        })}
      </span>
    </div>
  );
};
