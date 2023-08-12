"use client";
import { Box, Button, Grid, SxProps, TextField } from "@/app/components/atoms";
import { useForm, Controller } from "react-hook-form";
import * as yup from "yup";
import { yupResolver } from '@hookform/resolvers/yup';
import React from "react";

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

export const ConfigurableHookForm = ({
  fieldsData,
  onSubmitFormData,
  formSubmitType,
  containerStyle,
  onValueChange
}: Props) => {
  const { initalValues, validations } = configureParams(fieldsData);


  const { handleSubmit,  control, watch } = useForm({
    defaultValues: initalValues,
    resolver: yupResolver(yup.object(validations)),
  
  });

  const watchAllFields = watch(); 

  React.useEffect(() => {
    const subscription = watch((value, { name, type }) => {
      console.log(value, name, type);
      onValueChange
    });
    return () => subscription.unsubscribe();
  }, [watch, onValueChange]);

  // const formik = useFormik({
  //   initialValues: initalValues,
  //   validationSchema: yup.object(validations),
  //   onSubmit: (values, { setSubmitting }) => {
  //     onSubmitFormData(values, { setSubmitting });
  //   },
  //   validate: (values: any) => {
  //     onValueChange?.(values)
  //   },
  // });

  const onSubmit = (data: any) => {
        onSubmitFormData(data);
  };

  return (
    <Box
      sx={containerStyle || {}}
      
    >
      <form onSubmit={handleSubmit(onSubmit)} style={{width: '100%'}}>
        <Grid container columnSpacing={2} justifyContent={'space-between'}>
          {fieldsData.map((fieldData: ConfigurableFormFieldDataType) => {
            const grid = fieldData.grid || { xs: 12, md: 12, lg: 6, xl: 6 };
            return (
              <Grid item key={fieldData.id} {...grid}>
                <FieldRender fieldData={fieldData} control={control} />
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
  control,
}: {
  fieldData: ConfigurableFormFieldDataType;
  control: any;
}) => {




  if (fieldData.fieldType === "text-input")
    return (
      <Controller 
      name={fieldData.id}
      control={control}
      render={({field, fieldState: {
        error, isTouched
      }}) => 
        <TextField
        key={`field_${fieldData.id}`}
        id={fieldData.id}
        // name={fieldData.id}
        label={fieldData.label}
        variant="standard"
        autoFocus
        sx={{
          width: "100%",
          margin: "20px 0px"
        }}
        type={fieldData.type}
        // value={formik.values[fieldData.id]}
        // onChange={formik.handleChange}
        error={
          Boolean(error?.message)
        }
        helperText={error?.message || ""}
        {...field}
      />

      }
      
      />

    );
};
