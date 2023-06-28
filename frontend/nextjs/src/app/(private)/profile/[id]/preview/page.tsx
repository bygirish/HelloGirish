import React from "react";

type Props = {
  params: {
    id: string;
  }
}

export default function PreviewProfile(props: Props) {
  console.log("props", props);
  const {
    id: profileId
  } = props.params;
  console.log("profileId", profileId);
  return <div style={{display: 'flex', width: '100%',height: '100%', border: '1px solid #000000'}}><div>PreviewProfile: {profileId}</div></div>;
}
