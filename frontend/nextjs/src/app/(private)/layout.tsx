'use client'

import { LeftSidebar, PrivateContent } from "@/app/components/molecules";
import { PrivateHeader } from "@/app/components/molecules";
import { useAuthContext } from "@/context/AuthContext";
import useNavigator from "@/navigation/navigator";

import { routePath } from "@/navigation/routes";
import { useRouter } from "next/navigation";
import React, { useEffect } from "react";

export default function Layout({ children }: { children: React.ReactNode }) {
  const { authData } = useAuthContext();
  const { push } =  useNavigator();
  const router = useRouter();
  
  const {
   userId, sessionToken
  } = authData;
 
  useEffect(() => {
    if(!userId || !sessionToken) {
      push(routePath.home.public);
    }
  }, [userId, sessionToken, push]);


  return (
    <div>
      <PrivateHeader />
      <div
        style={{
          width: "100%",
          height: '100%',
          display: "flex",
          alignItems: 'flex-start',
          justifyContent: "flex-start",
          border: "1px solid #000000",
        }}
      >
        <LeftSidebar />
        <PrivateContent>{children}</PrivateContent>        
      </div>
    </div>
  );
}
