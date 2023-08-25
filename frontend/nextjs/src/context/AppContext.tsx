"use client";

import { useContext, useState } from "react";
import {
  AuthContext,
  AuthContextProvider,
  AuthContextState,
} from "./AuthContext";
import { ThemeProvider } from "@mui/material";
import { theme } from "@/theme/theme";
import {
  QueryClientProvider,
  QueryClient,
  // HydrationBoundary,
  dehydrate,
} from "react-query";
import getQueryClient from "@/lib/react-query/getQueryClient";

export const AppContextProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [client] = useState(
    new QueryClient({
      defaultOptions: {
        queries: {
          refetchOnWindowFocus: false,
          // refetchOnmount: false,
          refetchOnReconnect: false,
          retry: 1,
          staleTime: 5000,
        },
      },
    })
  );

  return (
    <QueryClientProvider client={client}>
      <ThemeProvider theme={theme}>
        <AuthContextProvider>{children}</AuthContextProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};
