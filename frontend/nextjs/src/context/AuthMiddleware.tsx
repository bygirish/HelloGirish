// import { useCookies } from "react-cookie";

import React from "react";
import { useAuthContext } from "./AuthContext";

type AuthMiddlewareProps = {
  children: React.ReactElement;
};

const AuthMiddleware: React.FC<AuthMiddlewareProps> = ({ children }) => {
  // const [cookies] = useCookies(["logged_in"]);
  const stateContext = useAuthContext();

  // if (query.isLoading && cookies.logged_in) {
  //   return <FullScreenLoader />;
  // }

  return children;
};

export default AuthMiddleware;
