// 'use client';

import React from "react";
import { useEffect, useRef, useState } from "react";
import EditorJS from "@editorjs/editorjs";
import Header from "@editorjs/header";
import List from "@editorjs/list";
import Quote from "@editorjs/quote";
import Delimiter from "@editorjs/delimiter";
import InlineCode from "@editorjs/inline-code";
import Marker from "@editorjs/marker";
import Embed from "@editorjs/embed";
import Image from "@editorjs/image";
import Table from "@editorjs/table";
import Warning from "@editorjs/warning";
import Code from "@editorjs/code";
import Checklist from "@editorjs/checklist";
import LinkTool from "@editorjs/link";
import Raw from "@editorjs/raw";
import Paragraph from "@editorjs/paragraph";
import Codebox from "@bomdi/codebox";


import { Box, Button } from "@/app/components/atoms";

import { Upload } from "upload-js"

const upload = Upload({ apiKey: "public_W142iAa9XenEJJxKTthSWgC14s7T" }); // Your real API key.

// const filePath = "/uploads/example.jpg";
// const fileUrl  = upload.url(filePath);



// import gql from "graphql-tag";
// import apolloClient from "../lib/apolloClient";
export default function Editor() {
  
  const editorRef = useRef<any>(null);
  const [editorData, setEditorData] = useState(null);
  const initEditor = () => {
    const editor = new EditorJS({
      holderId: "editorjs-container",
      tools: {
        header: {
          class: Header,
          inlineToolbar: ["marker", "link"],
          config: {
            levels: [1, 2, 3, 4, 5, 6],
            placeholder: "Enter a header",
            defaultLevel: 1,
          },
          shortcut: "CMD+SHIFT+H",
        },
        image: {
          class: Image,
          config: {
            uploader: {
              uploadByFile(file: File){
                console.log("file before uploading", file, file.name);
                return upload.uploadFile(file).then((res) => {
                  console.log("result", res);
                  return {
                    success: 1,
                    file: {
                      url: res.fileUrl
                      // any other image data you want to store, such as width, height, color, extension, etc
                    }
                  };
                });
              },
            }
          }

        }, 
        code: Code,
        paragraph: {
          class: Paragraph,
          inlineToolbar: true,
        },
        raw: Raw,
        inlineCode: InlineCode,
        list: {
          class: List,
          inlineToolbar: true,
          shortcut: "CMD+SHIFT+L",
        },
        checklist: {
          class: Checklist,
          inlineToolbar: true,
        },
        quote: {
          class: Quote,
          inlineToolbar: true,
          config: {
            quotePlaceholder: "Enter a quote",
            captionPlaceholder: "Quote's author",
          },
          shortcut: "CMD+SHIFT+O",
        },
        warning: Warning,
        marker: {
          class: Marker,
          shortcut: "CMD+SHIFT+M",
        },
        delimiter: Delimiter,
        // inlineCode: {
        //   class: InlineCode,
        //   shortcut: "CMD+SHIFT+C",
        // },
        linkTool: LinkTool,
        embed: Embed,
        codebox: Codebox,
        table: {
          class: Table,
          inlineToolbar: true,
          shortcut: "CMD+ALT+T",
        },
      },
      // autofocus: true,
      placeholder: "Write your story...",
      data: {
        blocks: [
          {
            type: "header",
            data: {
              text: "New blog post title here....",
              level: 2,
            },
          },
          {
            type: "paragraph",
            data: {
              text: "Blog post introduction here....",
            },
          },
        ],
      },
      onReady: () => {
        console.log("Editor.js is ready to work!");
        // @ts-ignore
        editorRef.current = editor;
      },
      onChange: () => {
        console.log("Content was changed");
      },
      //   onSave: () => {
      //     console.log("Content was saved");
      //   },
    });
  };
  const handleSave = async () => {
    // 1. GQL mutation to create a blog post in Fauna
    //   const CREATE_POST = gql`
    //     mutation CreatePost($content: String!, $slug: String!) {
    //       createPost(data: {published: true, content: $content, slug: $slug}) {
    //         content
    //         slug
    //         published
    //       }
    //     }
    //   `;
    // 2. Get the content from the editor
    const outputData = await editorRef?.current?.save();
    // 3. Get blog title to create a slug
    for (let i = 0; i < outputData.blocks.length; i++) {
      if (
        outputData.blocks[i].type === "header" &&
        outputData.blocks[i].data.level === 2
      ) {
        var title = outputData.blocks[i].data.text;
        break;
      }
    }
    const slug = title
      .toLowerCase()
      .replace(/ /g, "-")
      .replace(/[^\w-]+/g, "");
    console.log("Slug", slug);
    //3. Pass the content to the mutation and create a new blog post
    //   const { data } = await apolloClient.mutate({
    //     mutation: CREATE_POST,
    //     variables: {
    //       content: JSON.stringify(outputData),
    //       slug: slug,
    //     },
    //   });
  };
  useEffect(() => {
    if (!editorRef.current) {
      initEditor();
    }
  }, []);
  return (
    <>
      <Box
        sx={{
          width: "100%",
          background: "#cccccc56",
          position: "relative",
        }}
      >
        <Box
          sx={{
            display: "flex",
            margin: "30px",
            justifyItems: 'space-between',
          }}
        >
          <Box
            id="editorjs-container"
            sx={{
              flex: 1,
              height: "100%",
              border: "1px ridge #ccc",
              padding: "30px 20px",
              overflowY: "scroll",
              background: "#ffffff",
            }}
          />

          <Box
            sx={{
              width: "30%",
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
              px: "20px",
            }}
          >
            <Box
              sx={{
                width: "100%",
                height: "40vh",
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                background: "#ffffff",
             
              }}
            ></Box>

            <Box
              sx={{
                width: "100%",
                height: "40vh",
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                background: "#ffffff",
                my: '20px'
              }}
            ></Box>
          </Box>
        </Box>

        <Box
          sx={{
            position: "fixed",
            bottom: 0,
            width: "100%",
            border: "1px ridge #ccc",
            background: "#ffffff",
            p: "10px 30px"
          }}
        >
          <Button
            variant={"contained"}
            onClick={handleSave}
            sx={{
              width: "fit-content",
            }}
          >
            {"Save"}
          </Button>
        </Box>
      </Box>
    </>
  );
}
