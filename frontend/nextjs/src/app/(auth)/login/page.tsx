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
import { useAuthContext } from "@/context/AuthContext";
import { routePath } from "@/navigation/routes";

type Props = {};

type LoginFormDataType = {
  emailId: string;
  password: string;
};

export default function Login(props: Props) {

  const {authData, setAuthDetails } = useAuthContext();


  const onSubmit = (data: LoginFormDataType) => {
    
    console.log("onSubmit", data, authData);

    const {
      emailId,
      password
    } = data;

    setAuthDetails({
      userId: emailId,
      sessionToken: 'token-'+Math.random()*10000,
    })

  };

  const onValueChange = (data: LoginFormDataType) => {
    console.log("onValueChange: ", data);
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
            fieldsData={loginFormData}
            formSubmitType="submit"
            containerStyle={{
              display: "flex",
              justifyContent: "center",
              width: "100%",
            }}
            onValueChange={onValueChange}
            onSubmitFormData={onSubmit}
            submitButtonText={'Login'}
          />

          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Typography variant="subtitle1">
              {"Don't have account? Please"}
            </Typography>
            <Button variant='text'>
              <Link href={routePath.auth.register}>{"Sign up"}</Link>
            </Button>
            
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

const loginFormData: ConfigurableFormFieldDataType[] = [
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
];
