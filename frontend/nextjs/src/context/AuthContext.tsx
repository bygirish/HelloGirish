"use client";

import {
  createContext,
  useContext,
  useState,
} from "react";

type AuthDataType = {
    userId: string | undefined;
    sessionToken: string | undefined;
};

export type AuthContextState = {
  userId?: string;
  sessionToken?: string;
  setAuthDetails?: (authData: AuthDataType) => void;
};

const initialAuthContextState = {
  userId: "undefined",
  sessionToken: "undefined",
};

export const AuthContext = createContext<AuthContextState>(
  initialAuthContextState
);

export const AuthContextProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [userId, setUserId] = useState<string | undefined>(initialAuthContextState.userId);
  const [sessionToken, setSessionToken] = useState<string | undefined>(initialAuthContextState.sessionToken);

  const setAuthDetails = ({
    userId,
    sessionToken
  }: AuthDataType) => {
    setUserId(userId);
    setSessionToken(sessionToken)
  }
  

  return (
    <AuthContext.Provider
      value={{
        userId,
        sessionToken,
        setAuthDetails
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};


export const useAuthContext = (): AuthContextState => useContext(AuthContext);

