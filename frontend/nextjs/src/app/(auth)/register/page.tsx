"use client";
import { Box, Button, Grid, Typography } from "@/app/components/atoms/index";
import * as yup from "yup";
import React, { useState } from "react";
import {
  ConfigurableFormFieldDataType,
  ConfigurableFormFieldTypes,
  ConfigurableHookForm,
} from "@/app/components/molecules/ConfigurableHookForm";
import Link from "next/link";
import { routePath } from "@/navigation/routes";
import { useAuthContext } from "@/context/AuthContext";

type Props = {};

type SignUpFormDataType = {
  name: string;
  emailId: string;
  password: string;
  confirm_password: string;
};

export default function Register(props: Props) {
  const { authData, setAuthDetails } = useAuthContext();
  const [formErrors, setFormErrors] = useState<any[]>([]);

  const onSubmit = (data: SignUpFormDataType) => {
    console.log("onSubmit", data, authData);

    const errors = validateFormData(data);
    if (errors.length === 0) {
      const { emailId } = data;
      setAuthDetails({
        userId: emailId,
        sessionToken: "token-" + Math.random() * 10000,
        isOnboarding: true,
      });
    } else {
      setFormErrors(errors);
    }
  };

  const onValueChange = (values: SignUpFormDataType) => {
    console.log(values);
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
            formId={`register-form`}
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
            customErrors={formErrors}
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

const validateFormData = (data: SignUpFormDataType) => {
  const { password, confirm_password } = data;
  const errors = [];

  if (password !== confirm_password) {
    errors.push({
      fieldId: "confirm_password",
      type: "custom",
      errorMessage: "Password does not match",
    });
  }

  return errors;
};

const registerFormData: ConfigurableFormFieldDataType[] = [
  {
    id: "Name",
    label: "Name",
    type: 'text',
    validation: yup
      .string()
      .required("Name is required"),
    fieldType: ConfigurableFormFieldTypes.textInput,
    grid: { xs: 12, md: 12, lg: 12, xl: 12 },
  },
  {
    id: "emailId",
    label: "Email Id",
    type: "email",
    initialValue: "",
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
    initialValue: "",
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
    initialValue: "",
    validation: yup
      .string()
      .min(8, "Password should be of minimum 8 characters length")
      .required("Confirm Password is required"),
    fieldType: ConfigurableFormFieldTypes.textInput,
    grid: { xs: 12, md: 12, lg: 12, xl: 12 },
  },
];
