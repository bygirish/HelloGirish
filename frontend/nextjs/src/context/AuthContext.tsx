"use client";
// https://www.js-craft.io/blog/using-react-context-nextjs-13
import { createContext, useContext, useState } from "react";

type AuthDataType = {
  userId?: string | undefined;
  sessionToken?: string | undefined;
  isOnboarding?: boolean;
};

export type AuthContextState = {
  authData: AuthDataType;
  setAuthDetails: (authData: AuthDataType) => void;
};

const initialAuthContextState = {
  authData: {
    userId: undefined,
    sessionToken: undefined,
  },
  setAuthDetails: () => {},
};

export const AuthContext = createContext<AuthContextState>(
  initialAuthContextState
);

export const AuthContextProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [authData, setAuthData] = useState<AuthDataType>(
    initialAuthContextState.authData
  );

  const setAuthDetails = (authData: AuthDataType) => {
    setAuthData && setAuthData((state) => ({
      ...state,
      ...authData,
    }));
  };

  return (
    <AuthContext.Provider
      value={{
        authData,
        setAuthDetails,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuthContext = (): AuthContextState => {
  const context = useContext(AuthContext);

  if (context) {
    return context;
  }

  throw new Error(`useStateContext must be used within a StateContextProvider`);
};
