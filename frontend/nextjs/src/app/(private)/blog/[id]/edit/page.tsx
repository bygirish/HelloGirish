import React from "react";

type Props = {
  params: {
    id: string;
  }
}

export default function EditBlog(props: Props) {
  console.log("props", props);
  const {
    id: blogId
  } = props.params;
  console.log("EditBlog", blogId);
  return <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%',height: '100%', border: '1px solid #000000' }}><div>Edit Blog: {blogId}</div></div>;
}

