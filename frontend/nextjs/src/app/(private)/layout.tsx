'use client'

import { LeftSidebar, PrivateContent } from "@/components/molecules";
import { PrivateHeader } from "@/components/molecules";
import { useAuthContext } from "@/context/AuthContext";
import Navigator from "@/navigation/navigator";

import { routePath } from "@/navigation/routes";
import { useRouter } from "next/navigation";
import React from "react";

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
      {/* <div>Private Layout</div> */}
      <PrivateHeader />
      <div
        style={{
          width: "100%",
          minHeight: '100vh',
          maxHeight: "100%",
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
