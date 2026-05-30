'use client'
import { PublicContent } from "@/components/molecules";
import { PublicHeader } from "@/components/molecules/PublicHeader";
import { useAuthContext } from "@/context/AuthContext";
import Navigator from "@/navigation/navigator";
import { routePath } from "@/navigation/routes";
import { useRouter } from "next/navigation";
import React from "react";

export default function Layout({ children }: { children: React.ReactNode }) {
  
  return (
    <div>
      <div
        style={{
          padding: '50px',
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
