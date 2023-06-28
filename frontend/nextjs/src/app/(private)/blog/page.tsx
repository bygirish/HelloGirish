import React from "react";

type Props = {
  params: {
    id: string;
  }
}

export default function BlogListing(props: Props) {
  console.log("props", props);
  return <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%',height: '100%', border: '1px solid #000000' }}><div>BlogListing</div></div>;
}

