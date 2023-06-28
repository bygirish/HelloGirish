import React from "react";

type Props = {
  params: {
    id: string;
  }
}

export default function PreviewBlog(props: Props) {
  console.log("props", props);
  const {
    id: blogId
  } = props.params;
  console.log("PreviewBlog", blogId);
  return <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%',height: '100%', border: '1px solid #000000' }}><div>Preview Blog: {blogId}</div></div>;
}

