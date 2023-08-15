"use client";
import { Box, Button, Grid, SxProps, TextField } from "@/app/components/atoms";
import { useForm, Controller } from "react-hook-form";
import * as yup from "yup";
import { yupResolver } from "@hookform/resolvers/yup";
import React from "react";

export const ConfigurableFormFieldTypes = {
  textInput: "text-input",
} as const;

type keys = keyof typeof ConfigurableFormFieldTypes;

export type ConfigurableFormFieldDataType = {
  id: string;
  label: string;
  type?: string;
  initialValue: string;
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
  formId?: string;
  fieldsData: ConfigurableFormFieldDataType[];
  onSubmitFormData?: (...args: any[]) => void;
  formSubmitType?: "submit" | "reset" | "button" | undefined;
  containerStyle?: SxProps;
  onValueChange?: (values: any) => void;
  submitButtonText?: string;
};

export const ConfigurableHookForm = ({
  formId,
  fieldsData,
  onSubmitFormData,
  formSubmitType,
  containerStyle,
  onValueChange,
  submitButtonText
}: Props) => {
  const { initialValues, validations } = configureParams(fieldsData);

  const { handleSubmit, control, watch } = useForm({
    defaultValues: initialValues,
    resolver: yupResolver(yup.object(validations)),
    values: initialValues
  });

  const watchAllFields = watch();

  React.useEffect(() => {
    const subscription = watch((value, { name, type }) => {
      console.log(value, name, type);
      onValueChange?.(value);
    });
    return () => subscription.unsubscribe();
  }, [watch, onValueChange]);

  const onSubmit = (data: any) => {
    onSubmitFormData?.(data);
  };

  return (
    <Box sx={containerStyle || {}} id={formId}>
      <form onSubmit={handleSubmit(onSubmit)} style={{ width: "100%" }} >
        <Grid container columnSpacing={2} justifyContent={"space-between"}>
          {fieldsData.map((fieldData: ConfigurableFormFieldDataType, index) => {
            const grid = fieldData.grid || { xs: 12, md: 12, lg: 6, xl: 6 };
            return (
              <Grid item key={`${formId}-${fieldData.id}-grid`} {...grid}>
                <FieldRender formId={formId} fieldData={fieldData} control={control} isFocused={index === 0}/>
              </Grid>
            );
          })}
        </Grid>
        {onSubmitFormData && (
          <Button
            variant="contained"
            type={formSubmitType}
            sx={{
              margin: "30px 0px",
              width: "fit-content",
            }}
            color="primary"
          >
            {submitButtonText ?? "Submit"}
          </Button>
        )}
      </form>
    </Box>
  );
};

const configureParams = (fieldsData: ConfigurableFormFieldDataType[]) => {
  const initialValues: any = {};
  const validations: any = {};

  fieldsData.forEach((fieldData: ConfigurableFormFieldDataType) => {
    if (Array.isArray(fieldData)) {
      const data = configureParams(fieldData);
      Object.assign(initialValues, data.initialValues);
      Object.assign(validations, data.validations);
    } else {
      initialValues[fieldData.id] = fieldData.initialValue;
      validations[fieldData.id] = fieldData.validation;
    }
  });

  return {
    initialValues,
    validations,
  };
};

const FieldRender = ({
  fieldData,
  control,
  formId,
  isFocused
}: {
  fieldData: ConfigurableFormFieldDataType;
  formId?: string;
  control: any;
  isFocused: boolean
}) => {
  if (fieldData.fieldType === "text-input")
    return (
      <Controller
        key={`${formId}-field_${fieldData.id}-controller`}
        name={fieldData.id}
        control={control}
        render={({ field, fieldState: { error, isTouched } }) => (
          <TextField
            key={`${formId}-field_${fieldData.id}`}
            id={fieldData.id}
            // name={fieldData.id}
            label={fieldData.label}
            variant="standard"
            autoFocus={isFocused}
            sx={{
              width: "100%",
              margin: "5px 0px",
            }}
            type={fieldData.type}
            // value={formik.values[fieldData.id]}
            // onChange={formik.handleChange}
            error={Boolean(error?.message)}
            helperText={error?.message || ""}
            {...field}
          />
        )}
      />
    );
};
