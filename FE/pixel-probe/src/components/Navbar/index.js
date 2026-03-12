import { Link } from "react-router-dom";
import "./navbar.css";

export default function Navbar() {
  return (
    <nav className="nav">
      <img src="/PixelProbeLogo.png" alt="Logo" className="logo" />
      <ul className="navLinks">
        <li><Link to="/">Home</Link></li>
        <li><Link to="/history">History</Link></li>
      </ul>
    </nav>
  );
}
