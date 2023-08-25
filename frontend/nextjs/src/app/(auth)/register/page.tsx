"use client";
import { Box, Button, Grid, Typography } from "@/app/components/atoms/index";
import * as yup from "yup";
import React from "react";
import {
  ConfigurableFormFieldDataType,
} from "../../components/molecules/ConfigurableForm";
import {
  ConfigurableFormFieldTypes,
  ConfigurableHookForm,
} from "@/app/components/molecules/ConfigurableHookForm";
import Link from "next/link";
import { routePath } from "@/navigation/routes";

type Props = {};

type SignUpFormDataType = {
  emailId: string;
  password: string;
  confirm_password: string;
};

export default function Register(props: Props) {
  const onSubmit = (data: SignUpFormDataType) => {
    console.log(data);
  };

  const onValueChange = (values: SignUpFormDataType) => {
    console.log(values);
    // const {
    //   emailid,
    //   confirm_password,
    //   password
    // } = values;
    // if(password != confirm_password) {
    //   console.log("password mismatch");
    // }
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
      <Typography variant="h2">{"Sign up"}</Typography>

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
          <ConfigurableHookForm
            formId={`login-form`}
            fieldsData={registerFormData}
            formSubmitType="submit"
            containerStyle={{
              display: "flex",
              justifyContent: "center",
              width: "100%",
            }}
            onValueChange={onValueChange}
            onSubmitFormData={onSubmit}
            submitButtonText={"Sign Up"}
          />

          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Typography variant="subtitle1">
              {"Already have account? Please"}
            </Typography>
            <Button variant="text">
              <Link href={routePath.auth.login}>{"Login"}</Link>
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

const registerFormData: ConfigurableFormFieldDataType[] = [
  {
    id: "emailId",
    label: "Email Id",
    type: "email",
    initialValue: "abc123@gmail.com",
    validation: yup
      .string()
      .email("Enter a valid email")
      .required("Email is required"),
    fieldType: ConfigurableFormFieldTypes.textInput,
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
    fieldType: ConfigurableFormFieldTypes.textInput,
    grid: { xs: 12, md: 12, lg: 12, xl: 12 },
  },
  {
    id: "confirm_password",
    label: "Confirm Password",
    type: "password",
    initialValue: "abcd1234",
    validation: yup
      .string()
      .min(8, "Password should be of minimum 8 characters length")
      .required("Confirm Password is required"),
    fieldType: ConfigurableFormFieldTypes.textInput,
    grid: { xs: 12, md: 12, lg: 12, xl: 12 },
  },
];
