"use client";
import Navigator from "@/navigation/navigator";
import { Box, Button, Grid, Typography } from "@/app/components/atoms/index";
import * as yup from "yup";
import React from "react";
import { Form, Formik, useFormik } from "formik";
import {
  ConfigurableForm,
  ConfigurableFormFieldDataType,
} from "../../components/molecules/ConfigurableForm";
// import { useMutation } from "@tanstack/react-query";


type Props = {};

export default function Login(props: Props) {


  const onSubmit = (values: any, { setSubmitting }: any) => {
    setTimeout(() => {
      alert(JSON.stringify(values, null, 2));
      //   setSubmitting(false);
    }, 400);
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Typography variant="h2">{"Login"}</Typography>

      <Grid
        container
        spacing={5}
        sx={{
          margin: "20px auto",
          width: "90%",
        }}
      >
        <Grid
          item
          xs={12}
          md={6}
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            justifyContent: "center",
          }}
        >
          <Typography variant="h3">{"Welcome to Blog Builder"}</Typography>
          {/* <Typography variant="h5">{"by Girish"}</Typography> */}
        </Grid>
        <Grid
          item
          xs={12}
          md={6}
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >

          {/* <ConfigurableForm
            fieldsData={loginFormData}
            onSubmitFormData={onSubmit}
            formSubmitType="submit"
            containerStyle={{
              display: 'flex',
              justifyContent: 'center',
              width: '50%'
            }}
          /> */}

          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Typography variant="subtitle1">
              {"Don't have account? Please"}
            </Typography>
            <Button variant="text" color="primary">
              {"Sign up"}
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

const loginFormData: ConfigurableFormFieldDataType[] = [
  {
    id: "emailid",
    label: "Email-id",
    type: "email",
    initialValue: "abc123@gmail.com",
    validation: yup
      .string()
      .email("Enter a valid email")
      .required("Email is required"),
    fieldType: "text-input",
    grid: { xs: 12, md: 12, lg: 12, xl: 12 },
  },
  {
    id: "password",
    label: "Password",
    type: "password",
    initialValue: "abcd1234",
    validation: yup
      .string()
      .min(8, "Password should be of minimum 8 characters length")
      .required("Password is required"),
    fieldType: "text-input",
    grid: { xs: 12, md: 12, lg: 12, xl: 12 },
  },
];
