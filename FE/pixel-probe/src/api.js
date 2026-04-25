const API_URL = "http://100.49.9.230:8000";


export async function fetchPreds() {
  const token = localStorage.getItem("token");

  const response = await fetch(`${API_URL}/api/predictions`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    throw new Error("Failed to fetch predictions");
  }

  return response.json();
}


export async function createPred(file) {
  const token = localStorage.getItem("token");

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/uploadfile`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create prediction: ${errorText}`);
  }

  return response.json();
}


export async function registerUser(nameString, emailString, passwordString) {
  const response = await fetch(`${API_URL}/user/auth/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      name: nameString,
      email: emailString,
      password: passwordString,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to register user: ${errorText}`);
  }

  return response.json();
}


export async function loginUser(emailString, passwordString) {
  const response = await fetch(`${API_URL}/user/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email: emailString,
      password: passwordString,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to login user: ${errorText}`);
  }

  return response.json();
}
