import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from "./components/Navbar";
import ImageUpload from './components/fileUpload';
import MyGallery from './components/imageGallery';
import LoginPage from './components/userLogin';

function App() {
  return (
    <Router>
      <Navbar />
      <div className="mainpage">
        <Routes>
          <Route
            path="/"
            element={
              <div className="homepage">
                <img
                  src="/PixelProbeLogo.png"
                  alt="PixelProbe Logo"
                  className="homepage-logo"
                />
                <h1>Welcome to PixelProbe</h1>
                <LoginPage />
              </div>
            }
          />

          <Route path="/history" element={<MyGallery />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
