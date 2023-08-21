import React from "react";
import {
  LoginButton,
  LogoutButton,
  ProfileButton,
  RegisterButton,
} from "@/app/components/atoms/buttons.component";

export default function AuthPage({ children }: { children: React.ReactNode }) {
  return (
    <>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: "100%",
          height: "100%",
          border: "1px solid #000000",
        }}
      >
        <div>Welcome to Blog Builder</div>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: 'column',
          width: "30%",
        }}
      >
        <LoginButton />
        <RegisterButton />
        <LogoutButton />
        <ProfileButton />
      </div>
    </>
  );
}
