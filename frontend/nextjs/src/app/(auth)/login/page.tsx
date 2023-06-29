import Navigator from "@/navigation/navigator";
import React from "react";

export default function Login({ children }: { children: React.ReactNode }) {
  // console.log("Navigator.currentPathName", Navigator().currentPathName);
  return <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid #000000' }}><div>Login Page</div>{children}</div>;
}