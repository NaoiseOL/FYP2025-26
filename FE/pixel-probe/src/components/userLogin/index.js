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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from "@mui/material";
import {
  useSigninUserMutation,
  useLoginUserMutation,
} from "../../slices/loginSliceApi";
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
  const [fullName, setFullName] = useState(""); // For sign-up only
  const [userType, setUserType] = useState(""); // For user type selection
  const [signinUser] = useSigninUserMutation();
  const [loginUser] = useLoginUserMutation();
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    if (isSignUp) {
      console.log("Signing Up", { email, password, fullName, userType });
      const isSignupuser = await signinUser({
        name: fullName,
        email,
        password,
        userType,
      });
      console.log("isSignupuser", isSignupuser);
      if (isSignupuser?.data?.token) {
        localStorage.setItem("authToken", isSignupuser?.data?.token);
        localStorage.setItem("userType", userType); // Store user type in local storage
        if (userType === "interviewer") {
          navigate("/interviewer"); // Redirect to interviewer home page
        } else {
          navigate("/interviewee"); // Redirect to interviewee home page
        }
      }
    } else {
      console.log("Logging In", { email, password });
      const isLoginuser = await loginUser({ email, password });
      console.log("isLoginuser", isLoginuser);
      if (isLoginuser?.data?.token) {
        localStorage.setItem("authToken", isLoginuser?.data?.token);
        localStorage.setItem("userType", isLoginuser?.data?.user_type); // Store user type in local storage
        if (userType === "interviewer") {
          navigate("/interviewer"); // Redirect to interviewer home page
        } else {
          navigate("/interviewee"); // Redirect to interviewee home page
        }
      }
    }
  }

  const toggleForm = () => {
    setIsSignUp(!isSignUp);
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
        <Box component="form" noValidate autoComplete="off" sx={{ mt: 2 }}>
          {isSignUp && (
            <>
              <TextField
                label="Full Name"
                variant="outlined"
                fullWidth
                required
                margin="normal"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
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
              <FormControl fullWidth margin="normal">
                <InputLabel>User Type</InputLabel>
                <Select
                  value={userType}
                  onChange={(e) => setUserType(e.target.value)}
                  required
                >
                  <MenuItem value="interviewee">Interviewee</MenuItem>
                  <MenuItem value="interviewer">Interviewer</MenuItem>
                </Select>
              </FormControl>
            </>
          )}
          {!isSignUp && (
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
          )}
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
              if (password.length < 6) {
                console.log("Password must be at least 6 characters");
                return;
              }
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
