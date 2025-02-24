import React from "react";

const SignUp = () => {
  return (
    <div style={{ textAlign: "center", padding: "20px", color: "white" }}>
      <h2>Sign Up</h2>
      <form>
        <input type="text" placeholder="Username" required style={{ padding: "10px", margin: "10px 0" }} />
        <br />
        <input type="email" placeholder="Email" required style={{ padding: "10px", margin: "10px 0" }} />
        <br />
        <input type="password" placeholder="Password" required style={{ padding: "10px", margin: "10px 0" }} />
        <br />
        <button type="submit" style={{ padding: "10px 20px", cursor: "pointer" }}>Sign Up</button>
      </form>
    </div>
  );
};

export default SignUp;