"use client";
import Navigator from "@/navigation/navigator";
import { Button } from "@mui/material";
// import puppeteer from "puppeteer";
import React from "react";

type Props = {};


export default function Dashboard(props: Props) {
  // console.log("props", props, Navigator().currentPathName);
  return (
    <div
      style={{
        display: "flex",
        width: "100%",
        height: "100%",
        border: "1px solid #000000",
      }}
    >
      <div>Dashboard</div>
      <Button
        onClick={() => {
          printPDF();
        }}
      >
        Print PDF
      </Button>
    </div>
  );
}
