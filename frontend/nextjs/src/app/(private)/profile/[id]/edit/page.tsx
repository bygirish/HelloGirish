"use client";
import Navigator from "@/navigation/navigator";
import { Box, Button, Grid, Typography } from "@/app/components/atoms/index";
import * as yup from "yup";
import React from "react";
import { Form, Formik, useFormik } from "formik";
import {
  ConfigurableForm,
  ConfigurableFormFieldDataType,
} from "@/app/components/molecules/ConfigurableForm";

type Props = {
  params: {
    id: string;
  }
}

export default function EditProfile(props: Props) {
  const onSubmit = (values: any, { setSubmitting }: any) => {
    setTimeout(() => {
      alert(JSON.stringify(values, null, 2));
      //   setSubmitting(false);
    }, 400);
  };

  const {
    id: profileId
  } = props.params;

  const onValueChange = (values: any) => {
    console.log(values);
    const {
      emailid,
      confirm_password,
      password
    } = values;
    if(password != confirm_password) {
      console.log("password mismatch");
    }
  }

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Typography variant="h2">{"Edit Profile"}</Typography>
      <ConfigurableForm
            fieldsData={profileFormData()}
            onSubmitFormData={onSubmit}
            formSubmitType="submit"
            containerStyle={{
              display: 'flex',
              justifyContent: 'center',
              width: '50%'
            }}
            onValueChange={onValueChange}
          />
    </Box>
  );
}

const profileFormData = (initialData?: any): ConfigurableFormFieldDataType[] => {

  const {
    firstName,
    lastName,
    gender,
    dob,
    location,
  } = initialData || {};


  return (
    [
      {
        id: "firstName",
        label: "First Name",
        initalValue: firstName || '',
        validation: yup
          .string()
          .required("First Name is required"),
        fieldType: "text-input",
        grid: {xs: 12, md: 12, lg: 6, xl: 6}
      },
      {
        id: "lastName",
        label: "Last Name",
        initalValue: lastName || '',
        validation: yup
          .string()
          .required("Last Name is required"),
        fieldType: "text-input",
        grid: {xs: 12, md: 12, lg: 6, xl: 6}
      },
      {
        id: "gender",
        label: "Gender",
        initalValue: gender || '',
        validation: yup
          .string()
          .oneOf(['male', 'female'], 'Please specify a valid gender')
          .required("Gender is required"),
        fieldType: "text-input",
        grid: {xs: 12, md: 12, lg: 12, xl: 12}
      },
      {
        id: "dob",
        label: "Date of Birth",
        initalValue: dob || '',
        validation: yup
          .date()
          .required("Date of Birth is required"),
        fieldType: "text-input",
        grid: {xs: 12, md: 12, lg: 6, xl: 6}
      },
      {
        id: "location",
        label: "Location",
        initalValue: location || '',
        validation: yup
          .string()
          .required("Location is required"),
        fieldType: "text-input",
        grid: {xs: 12, md: 12, lg: 6, xl: 6}
      },
    ]
  );
}

