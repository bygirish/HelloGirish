"use client";
import { Box, Button, Grid, SxProps, TextField } from "@/app/components/atoms";
import { useFormik } from "formik";
import * as yup from "yup";

export const ConfigurableFormFieldTypes = {
  textinput: "text-input",
} as const;

type keys = keyof typeof ConfigurableFormFieldTypes;

export type ConfigurableFormFieldDataType = {
  id: string;
  label: string;
  type?: string;
  initalValue: string;
  validation: any; //StringSchema<string, AnyObject, undefined, "">; //(...args: any[]) => void;
  fieldType: (typeof ConfigurableFormFieldTypes)[keys];
  grid: {
    xs?: number;
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
  };
};

type Props = {
  fieldsData: ConfigurableFormFieldDataType[];
  onSubmitFormData: (...args: any[]) => void;
  formSubmitType?: "submit" | "reset" | "button" | undefined;
  containerStyle?: SxProps;
  onValueChange?: (values: any) => void;
};

export const ConfigurableForm = ({
  fieldsData,
  onSubmitFormData,
  formSubmitType,
  containerStyle,
  onValueChange
}: Props) => {
  const { initalValues, validations } = configureParams(fieldsData);

  const formik = useFormik({
    initialValues: initalValues,
    validationSchema: yup.object(validations),
    onSubmit: (values, { setSubmitting }) => {
      onSubmitFormData(values, { setSubmitting });
    },
    validate: (values: any) => {
      onValueChange?.(values)
    },
  });

  return (
    <Box
      sx={containerStyle || {}}
      
    >
      <form onSubmit={formik.handleSubmit} style={{width: '100%'}}>
        <Grid container columnSpacing={2}>
          {fieldsData.map((fieldData: ConfigurableFormFieldDataType) => {
            const grid = fieldData.grid || { xs: 12, md: 12, lg: 6, xl: 6 };
            return (
              <Grid item key={fieldData.id} {...grid}>
                <FieldRender fieldData={fieldData} formik={formik} />
              </Grid>
            );
          })}
        </Grid>
        <Button
          variant="contained"
          type={formSubmitType}
          sx={{
            margin: "30px 0px",
            width: "fit-content",
          }}
          color="primary"
        >
          {"Login"}
        </Button>
      </form>
    </Box>
  );
};

const configureParams = (fieldsData: ConfigurableFormFieldDataType[]) => {
  const initalValues: any = {};
  const validations: any = {};

  fieldsData.forEach((fieldData: ConfigurableFormFieldDataType) => {
    if (Array.isArray(fieldData)) {
      const data = configureParams(fieldData);
      Object.assign(initalValues, data.initalValues);
      Object.assign(validations, data.validations);
    } else {
      initalValues[fieldData.id] = fieldData.initalValue;
      validations[fieldData.id] = fieldData.validation;
    }
  });

  return {
    initalValues,
    validations,
  };
};

const FieldRender = ({
  fieldData,
  formik,
}: {
  fieldData: ConfigurableFormFieldDataType;
  formik: any;
}) => {
  if (fieldData.fieldType === "text-input")
    return (
      <TextField
        key={`field_${fieldData.id}`}
        id={fieldData.id}
        name={fieldData.id}
        label={fieldData.label}
        variant="standard"
        autoFocus
        sx={{
          width: "100%",
          margin: "20px 0px"
        }}
        type={fieldData.type}
        value={formik.values[fieldData.id]}
        onChange={formik.handleChange}
        error={
          formik.touched[fieldData.id] && Boolean(formik.errors[fieldData.id])
        }
        helperText={
          formik.touched[fieldData.id]
            ? (formik.errors[fieldData.id] as string)
            : ""
        }
      />
    );
};
