import React from "react";
import "keen-slider/keen-slider.min.css";
import { useKeenSlider } from "keen-slider/react";
import "./imageGallery.css"

export default function App() {
  const [sliderRef] = useKeenSlider({
    loop: true,
    mode: "snap",
    slides: {
      perView: 1,
      spacing: 15,
    },
    breakpoints: {
      "(min-width: 640px)": {
        slides: { perView: 1, spacing: 15 },
      },
      "(min-width: 1024px)": {
        slides: { perView: 1, spacing: 20 },
      },
    },
  });


  return (
    <div className="App">
      <h2>Keen Slider React Example</h2>
      <div ref={sliderRef} className="keen-slider">
        <div className="keen-slider__slide number-slide1">
          <div className="homepage-logo-wrapper">
            <img 
              src="/PixelProbeLogo.png" 
              alt="PixelProbe Logo" 
              className="homepage-logo" 
              width={224}
              height={224}
            />
            <div className="logo-text">PixelProbeLogo</div>
          </div>
        </div>

        <div className="keen-slider__slide number-slide2">Slide 2</div>
        <div className="keen-slider__slide number-slide3">Slide 3</div>
        <div className="keen-slider__slide number-slide4">Slide 4</div>
        <div className="keen-slider__slide number-slide5">Slide 5</div>
      </div>
    </div>
  );
}
