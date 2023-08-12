"use client";
// https://www.js-craft.io/blog/using-react-context-nextjs-13
import {
  createContext,
  useContext,
  useState,
} from "react";

type AuthDataType = {
    userId?: string | undefined;
    sessionToken?: string | undefined;
};

export type AuthContextState = {
  userId?: string;
  sessionToken?: string;
  setAuthDetails?: (authData: AuthDataType) => void;
};

const initialAuthContextState = {
  userId: '123', //undefined,
  sessionToken: '234', // undefined,
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
    userId && setUserId(userId);
    sessionToken && setSessionToken(sessionToken)
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

