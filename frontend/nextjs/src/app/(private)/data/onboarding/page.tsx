"use client";
import Navigator from "@/navigation/navigator";
import {
  Box,
  Button,
  Grid,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from "@/app/components/atoms/index";
import * as yup from "yup";
import React from "react";
import { Form, Formik, useFormik } from "formik";
import {
  ConfigurableForm,
  ConfigurableFormFieldDataType,
} from "@/app/components/molecules/ConfigurableForm";
import { ConfigurableHookForm } from "@/app/components/molecules/ConfigurableHookForm";
import ProfileDataElement from "../components/ProfileDataElement";
import { ProfileDataElements } from "@/modules/data/constants";

type Props = {
  params: {
    id: string;
  };
};

const onboardingSteps = [
  {
    key: ProfileDataElements.personal,
    text: "Personal Information",
  },
  {
    key: ProfileDataElements.education,
    text: "Education",
  },
  {
    key: ProfileDataElements.workExperience,
    text: "Work Experience",
  },
  {
    key: ProfileDataElements.certificates,
    text: "Certificates",
  },
  {
    key: ProfileDataElements.references,
    text: "References",
  },
  {
    key: ProfileDataElements.hobbies,
    text: "Hobbies",
  },
];

export default function OnboardingData(props: Props) {
  const activeStep = 1;

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        alignContent: "center",
        width: "100%",
      }}
    >
      <Typography variant="h3">{"Complete Profile"}</Typography>
      <Stepper
        activeStep={activeStep}
        alternativeLabel
        sx={{
          width: "50%",
          my: '50px'
        }}
      >
        {onboardingSteps.map((stepData) => {
          return (
            <Step key={stepData.key}>
              <StepLabel>{stepData.text}</StepLabel>
            </Step>
          );
        })}
      </Stepper>
      <Box
        sx={{
          width: "50%",
        }}
      >
        <ProfileDataElement dataType={onboardingSteps[activeStep].key} />
      </Box>
    </Box>
  );
}

