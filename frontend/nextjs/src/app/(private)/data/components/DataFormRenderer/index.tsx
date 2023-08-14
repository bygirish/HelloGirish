"use client";
import { Box } from "@/app/components/atoms/index";
import React, { useState } from "react";
import {
  ConfigurableFormFieldDataType,
  ConfigurableHookForm,
} from "@/app/components/molecules/ConfigurableHookForm";
import { MultiElementsContainer } from "@/app/components/molecules/MutiElementsContainer.tsx";

type Props = {
  formId: string;
  configurationData: ConfigurableFormFieldDataType[];
  isMultipleForms: boolean;
  titleElementMultipleForms?: string;
};

export default function DataFormRenderer(props: Props) {
  const {
    formId,
    isMultipleForms,
    titleElementMultipleForms,
    configurationData,
  } = props;

  const [elementDataSingleForm, setElementDataSingleForm] = useState([{}]);
  const [elementsDataMultipleForms, setElementsDataMultipleForms] = useState([
    {},
  ]);

  if (!isMultipleForms) {
    const onValueChange = (updatedData: any) => {
      setElementDataSingleForm(updatedData);
    };

    return (
      <ConfigurableHookForm
        formId={formId}
        fieldsData={configurationData}
        formSubmitType="submit"
        containerStyle={{
          display: "flex",
          justifyContent: "center",
          m: "10px",
          p: "5px 10px",
          border: "1px ridge #ccc",
          background: "#cccccc30",
        }}
        onValueChange={onValueChange}
      />
    );
  }

  return (
      <MultiElementsContainer
        titleElement={titleElementMultipleForms}
        formId={formId}
        data={elementsDataMultipleForms}
        onDataUpdate={(data) => {
          setElementsDataMultipleForms(data);
        }}
      >
        {elementsDataMultipleForms.map((data: any, index) => {
          const onValueChange = (updatedData: any) => {
            setElementsDataMultipleForms((data) => {
              const newData = [...data];
              newData[index] = updatedData;
              return newData;
            });
          };
          return (
            <ConfigurableHookForm
              key={index}
              formId={`${formId}-${index}`}
              fieldsData={configurationData}
              formSubmitType="submit"
              containerStyle={{
                display: "flex",
                justifyContent: "center",
                width: '100%'
              }}
              onValueChange={onValueChange}
            />
          );
        })}
      </MultiElementsContainer>
  );
}
