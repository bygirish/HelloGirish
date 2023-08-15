import CreateBlog from "@/app/components/molecules/CustomEditor/CreateBlog";
import React from "react";

type Props = {
  params: {
    id: string;
  };
};

export default function EditBlog(props: Props) {
  console.log("props", props);
  const { id: blogId } = props.params;
  console.log("EditBlog", blogId);
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        border: "1px solid #000000",
        padding: '20px'
      }}
    >
      {/* <div>Edit Blog: {blogId}</div> */}
      <CreateBlog />
    </div>
  );
}
