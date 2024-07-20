"use client";
import Navigator from "@/navigation/navigator";
import { Box, Button, Grid, Step, StepLabel, Stepper, Typography } from "@/app/components/atoms/index";
import * as yup from "yup";
import React, { useState } from "react";
import { Form, Formik, useFormik } from "formik";
import {
  ConfigurableForm,
  ConfigurableFormFieldDataType,
} from "@/app/components/molecules/ConfigurableForm";
import { ConfigurableHookForm } from "@/app/components/molecules/ConfigurableHookForm";

type Props = {
  params: {
    id: string;
  };
};



export default function EditProfile(props: Props) {


  const [currentStep, setCurrentStep] = useState<number>(0);

  const [capturedFormData, setCapturedFormData] = useState<any[]>([]);

  const onValueChange = (values: any) => {
    console.log(values);
    const { emailid, confirm_password, password } = values;
    if (password != confirm_password) {
      console.log("password mismatch");
    }
  };

  const onSubmit = (values: any) => {

    setTimeout(() => {
      console.log("values", values);
      // alert(JSON.stringify(values, null, 2));
      //   setSubmitting(false);
      if(currentStep < steps.length) {
        setCapturedFormData([...capturedFormData, values]);
        setCurrentStep(currentStep+1);
      }
    }, 400);
  };

  console.log(JSON.stringify(capturedFormData, null));
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
      {/* <ConfigurableForm
        fieldsData={profileFormData()}
        onSubmitFormData={onSubmit}
        formSubmitType="submit"
        containerStyle={{
          display: "flex",
          justifyContent: "center",
          width: "50%",
        }}
        onValueChange={onValueChange}
      /> */}

      <Stepper activeStep={currentStep} alternativeLabel sx={{
        width: "100%",
      }}>
        {steps.map((stepData) => {
          console.log("stepData", stepData);

          return (
            <Step key={stepData.key}>
              <StepLabel>{stepData.text}</StepLabel>
            </Step>
          )
        })
        } 
      </Stepper>

      <ConfigurableHookForm
        key={steps[currentStep].key}
        formId={steps[currentStep].key}
        fieldsData={steps[currentStep].data}
        onSubmitFormData={onSubmit}
        formSubmitType="submit"
        containerStyle={{
          display: "flex",
          justifyContent: "center",
          width: "50%",
        }}
        onValueChange={onValueChange}
      />
    </Box>
  );
}




const profileFormData = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const { firstName, lastName, gender, dob, location } = initialData || {};

  return [
    {
      id: "firstName",
      label: "First Name",
      initialValue: firstName || "",
      validation: yup.string().required("First Name is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "lastName",
      label: "Last Name",
      initialValue: lastName || "",
      validation: yup.string().required("Last Name is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "gender",
      label: "Gender",
      initialValue: gender || "",
      validation: yup
        .string()
        .oneOf(["male", "female"], "Please specify a valid gender")
        .required("Gender is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 12, xl: 12 },
    },
    {
      id: "dob",
      label: "Date of Birth",
      initialValue: dob || "",
      // validation: yup.date().required("Date of Birth is required"),\
      validation: yup.string().required("Date of Birth is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "location",
      label: "Location",
      initialValue: location || "",
      validation: yup.string().required("Location is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    
  ];
};


const educationFormData = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const { college, location, course, score,  } = initialData || {};

  return [
    {
      id: "college",
      label: "College Name",
      initialValue: college || "",
      validation: yup.string().required("College Name is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 12, xl: 12 },
    },
    {
      id: "location",
      label: "Location",
      initialValue: location || "",
      validation: yup.string().required("Location is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "course",
      label: "Course Name",
      initialValue: course || "",
      validation: yup
        .string()
        .required("Course Name is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 12, xl: 12 },
    },
    {
      id: "score",
      label: "Score",
      initialValue: score || "",
      validation: yup.string().required("Score is required"),
      fieldType: "text-input",
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    
  ];
};

const steps = [
  {
    key: 'profile',
    text: 'Profile',
    data: profileFormData(),
  },
  {
    key: 'education',
    text: 'Education',
    data: educationFormData(),
  },
  {
    key: 'work',
    text: 'Work',
    data: profileFormData(),
  },
]