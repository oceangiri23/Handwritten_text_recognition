import React, { useEffect } from "react";
import { useAuthState } from "react-firebase-hooks/auth";
import { useHistory } from "react-router-dom";
import { auth, signInWithGoogle } from "../firebase";
import logo from "../images/logo.png";
import "./Home.css";

function Home() {
  const [user, loading, error] = useAuthState(auth);
  const history = useHistory();

  useEffect(() => {
    if (loading) return;
    if (user) history.push("/dashboard");
  }, [loading, user]);

  return (
    <div className="home">
      <div className="home__container">
        <img
          src={logo}
          alt="Scan-grade Logo"
          className="home__image"
          style={{ width: "300px" }}
        />
        <button className="home__login" onClick={signInWithGoogle}>
          Login with Google
        </button>
      </div>
    </div>
  );
}

export default Home;
