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

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#90caf9",
    },
    background: {
      paper: "#121212",
      default: "#202020",
    },
    text: {
      primary: "#ffffff",
    },
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

  const toggleForm = () => {
    setIsSignUp(!isSignUp);
    setError("");
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container
        maxWidth="xs"
        sx={{
          mt: 8,
          p: 4,
          borderRadius: 2,
          boxShadow: 3,
          bgcolor: "background.paper",
        }}
      >
        <Typography
          variant="h4"
          component="h1"
          gutterBottom
          align="center"
          color="text.primary"
        >
          {isSignUp ? "Sign Up" : "Login"}
        </Typography>

        {error && (
          <Typography color="error" align="center" sx={{ mb: 1 }}>
            {error}
          </Typography>
        )}

        <Box component="form" noValidate autoComplete="off" sx={{ mt: 2 }}>
          {isSignUp && (
            <TextField
              label="Full Name"
              variant="outlined"
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
            variant="outlined"
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
            variant="outlined"
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
            color="primary"
            fullWidth
            sx={{ mt: 2 }}
            onClick={(e) => {
              e.preventDefault();
              if (password.length < 6) return;
              handleSubmit(e);
            }}
          >
            {isSignUp ? "Sign Up" : "Login"}
          </Button>
        </Box>

        <Box textAlign="center" sx={{ mt: 2 }}>
          <Link
            component="button"
            variant="body2"
            onClick={toggleForm}
            sx={{ textDecoration: "underline" }}
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
