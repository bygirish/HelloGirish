import React from "react";

type Props = {}

export default function Dashboard(props: Props) {
  console.log("props", props);
  return <div style={{display: 'flex', width: '100%',height: '100%', border: '1px solid #000000' }}><div>Dashboard</div></div>;
}

