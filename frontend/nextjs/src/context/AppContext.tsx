"use client";

import { useContext } from "react";
import {
  AuthContext,
  AuthContextProvider,
  AuthContextState,
} from "./AuthContext";
import { ThemeProvider } from "@mui/material";
import { theme } from "@/theme/theme";

export const AppContextProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  return (
    <ThemeProvider theme={theme}>
      <AuthContextProvider>{children}</AuthContextProvider>
    </ThemeProvider>
  );
};
