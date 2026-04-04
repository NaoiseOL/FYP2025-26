import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import axios from "axios";

export const loginSliceApi = createApi({
  reducerPath: "loginApi",
  baseQuery: fetchBaseQuery({ baseUrl: "/" }),
  endpoints: (builder) => ({
    signinUser: builder.mutation({
      queryFn: _signinUser,
    }),
    loginUser: builder.mutation({
      queryFn: _loginUser,
    }),
    logoutUser: builder.mutation({
      query: () => ({
        url: "/logout",
        method: "POST",
      }),
    }),
  }),
});

async function _signinUser({ name, email, password, userType }) {
  const body = {
    name: name,
    email: email,
    password: password,
    user_type: userType,
  };
  try {
    console.log("Registering user:", { name, email, password });
    const response = await axios.post(
      "http://localhost:8000/user/register",
      body,
      {
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    const data = await response.data;
    console.log("User registered:", data);
    return { data }; // Return the response data
  } catch (err) {
    console.error("Registration failed", err);
  }
}

async function _loginUser({ email, password }) {
  const body = {
    email: email,
    password: password,
  };
  try {
    console.log("Logging in user:", { email, password });
    const response = await axios.post(
      "http://localhost:8000/user/login",
      body,
      {
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    const data = await response.data;
    console.log("User logged in:", data);
    return { data }; // Return the response data
  } catch (err) {
    console.error("Login failed", err);
  }
}

export const {
  useSigninUserMutation,
  useLogoutUserMutation,
  useLoginUserMutation,
} = loginSliceApi;
