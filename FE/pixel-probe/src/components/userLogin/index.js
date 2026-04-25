import React, { useState } from "react";
import {
  Container,
  Typography,
  TextField,
  Button,
  Box,
  Link,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from "@mui/material";
import { registerUser, loginUser } from "../../api";
import { useNavigate } from "react-router-dom";
import "./login.css";

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#90caf9" },
    background: {
      paper: "#121212",
      default: "#202020",
    },
    text: { primary: "#ffffff" },
  },
});

const LoginPage = () => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");

    try {
      if (isSignUp) {
        const data = await registerUser(fullName, email, password);
        if (data?.access_token) {
          localStorage.setItem("token", data.access_token);
          navigate("/predictions");
        }
      } else {
        const data = await loginUser(email, password);
        if (data?.access_token) {
          localStorage.setItem("token", data.access_token);
          navigate("/predictions");
        }
      }
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />

      <Container
        maxWidth="xs"
        className="login-container"
        sx={{ bgcolor: "background.paper" }}
      >
        <Typography variant="h4" align="center" gutterBottom>
          {isSignUp ? "Sign Up" : "Login"}
        </Typography>

        {error && (
          <Typography color="error" align="center" sx={{ mb: 1 }}>
            {error}
          </Typography>
        )}

        <Box
          component="form"
          noValidate
          autoComplete="off"
          className="login-form"
        >
          {isSignUp && (
            <TextField
              label="Full Name"
              fullWidth
              required
              margin="normal"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
          )}

          <TextField
            label="Email"
            type="email"
            fullWidth
            required
            margin="normal"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            error={
              email.length > 0 && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
            }
            helperText={
              email.length > 0 && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
                ? "Please enter a valid email address"
                : ""
            }
          />

          <TextField
            label="Password"
            type="password"
            fullWidth
            required
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            error={password.length > 0 && password.length < 6}
            helperText={
              password.length > 0 && password.length < 6
                ? "Password must be at least 6 characters"
                : ""
            }
          />

          <Button
            type="submit"
            variant="contained"
            fullWidth
            className="login-button"
            onClick={(e) => {
              e.preventDefault();
              if (password.length < 6) return;
              handleSubmit(e);
            }}
          >
            {isSignUp ? "Sign Up" : "Login"}
          </Button>
        </Box>

        <Box textAlign="center">
          <Link
            component="button"
            variant="body2"
            className="login-toggle"
            onClick={() => {
              setIsSignUp(!isSignUp);
              setError("");
            }}
          >
            {isSignUp
              ? "Already have an account? Login"
              : "Don't have an account? Sign Up"}
          </Link>
        </Box>
      </Container>
    </ThemeProvider>
  );
};

export default LoginPage;
