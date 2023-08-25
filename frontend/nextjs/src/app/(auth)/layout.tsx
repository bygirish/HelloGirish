'use client'
import { PublicContent } from "@/app/components/molecules";
import { PublicHeader } from "@/app/components/molecules/PublicHeader";
import { useAuthContext } from "@/context/AuthContext";
import useNavigator from "@/navigation/navigator";
import { routePath } from "@/navigation/routes";
import React, { useEffect } from "react";

export default function Layout({ children }: { children: React.ReactNode }) {
  const { authData } = useAuthContext();
  const { push } = useNavigator();
  
  const {
   userId, sessionToken
  } = authData;

  useEffect(() => {
    if(userId && sessionToken) {
      push(routePath.home.private);
     } 
  }, [userId, sessionToken, push]);
 

  return (
    <div>
      <PublicHeader />
      <div>Auth Layout</div>
      <div
        style={{
          width: "100%",
          minHeight: '100vh',
          maxHeight: "100%",
          display: "flex",
          alignItems: 'flex-start',
          justifyContent: "center",
          border: "1px solid #000000",
        }}
      >
        <PublicContent>{children}</PublicContent>
      </div>
    </div>
  );
}
