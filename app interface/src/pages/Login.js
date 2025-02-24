import React from "react";

const Login = () => {
  return (
    <div style={{ textAlign: "center", padding: "20px", color: "white" }}>
      <h2>Login</h2>
      <form>
        <input type="text" placeholder="Username" required style={{ padding: "10px", margin: "10px 0" }} />
        <br />
        <input type="password" placeholder="Password" required style={{ padding: "10px", margin: "10px 0" }} />
        <br />
        <button type="submit" style={{ padding: "10px 20px", cursor: "pointer" }}>Login</button>
      </form>
    </div>
  );
};

export default Login;
