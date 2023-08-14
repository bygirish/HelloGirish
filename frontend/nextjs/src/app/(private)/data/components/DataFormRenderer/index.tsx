"use client";
import { Box } from "@/app/components/atoms/index";
import React, { useState } from "react";
import {
  ConfigurableFormFieldDataType,
  ConfigurableHookForm,
} from "@/app/components/molecules/ConfigurableHookForm";
import { MultiElementsContainer } from "@/app/components/molecules/MutiElementsContainer.tsx";
import useProfileDataStore from "@/modules/data/store";
import { ProfileDataElementsType } from "@/modules/data/types";
import { getFormConfigurations } from "@/modules/data/configs";

type Props = {
  profileDataElementType: ProfileDataElementsType;
  isMultipleForms: boolean;
  titleElementMultipleForms?: string;
};

export default function DataFormRenderer(props: Props) {
  const {
    profileDataElementType,
    isMultipleForms,
    titleElementMultipleForms,
  } = props;

  const {
    profileData,
    updateProfileData
  } = useProfileDataStore((state: any) => {
    return {
      profileData: state.profileData,
      updateProfileData: state.updateProfileData
    }
  });





  const elementsData: any | any[] = profileData[profileDataElementType]

  if(!elementsData) {
    updateProfileData(profileDataElementType, isMultipleForms ? [{}] : {});
    return null;
  }



  const setElementData = (updatedData: any) => {
      console.log("updatedData", updatedData);
      updateProfileData(profileDataElementType, updatedData);
  }

  if (!isMultipleForms) {
    const configurationData = getFormConfigurations(profileDataElementType, profileData[profileDataElementType]);

    return (
      <ConfigurableHookForm
        formId={profileDataElementType}
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
        onValueChange={setElementData}
      />
    );
  }


  return (
      <MultiElementsContainer
        titleElement={titleElementMultipleForms}
        formId={profileDataElementType}
        data={elementsData}
        onDataUpdate={(data) => {
          setElementData(data);
        }}
      >
        {elementsData.map((data: any, index: number) => {


          const configurationData = getFormConfigurations(profileDataElementType, data);
          console.log("configurationData", configurationData);

          const onValueChange = (updatedData: any) => {
            const newData = [...elementsData];
            newData[index] = updatedData;
            setElementData(newData);
          };
          return (
            <ConfigurableHookForm
              key={index}
              formId={`${profileDataElementType}-${index}`}
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
