import React from "react";

export default function Layout({ children }: { children: React.ReactNode }) {
  return <div style={{width: '100%', height: '100vh'}}><div>Profile Layout</div><div>{children}</div></div>;
}
