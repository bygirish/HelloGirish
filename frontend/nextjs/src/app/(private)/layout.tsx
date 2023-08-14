'use client'

import { LeftSidebar, PrivateContent } from "@/app/components/molecules";
import { PrivateHeader } from "@/app/components/molecules";
import { useAuthContext } from "@/context/AuthContext";
import Navigator from "@/navigation/navigator";

import { routePath } from "@/navigation/routes";
import { useRouter } from "next/navigation";
import React from "react";
import { Box } from "@/app/components/atoms";

export default function Layout({ children }: { children: React.ReactNode }) {
  const authDetails = useAuthContext();
  const router = useRouter();
  
  const {
   userId, sessionToken
  } = authDetails;
 
  if(!userId || !sessionToken) {
    Navigator
    Navigator().push(routePath.home.public);
  }
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
