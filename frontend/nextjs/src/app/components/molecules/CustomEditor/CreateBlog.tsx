"use client";

import dynamic from "next/dynamic";
import { Box, Typography } from "@/app/components/atoms";
const Editor = dynamic(() => import("./CustomEditor"), { ssr: false });
export default function CreateBlog() {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
      }}
    >
      <Typography
        variant="h2"
        sx={{
          mb: "30px",
        }}
      >
        {"Create Blog"}
      </Typography>
      <Editor />
    </Box>
  );
}
