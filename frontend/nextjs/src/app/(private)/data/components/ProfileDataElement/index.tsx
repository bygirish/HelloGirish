import { ProfileDataElements } from "@/modules/data/constants";
import { ProfileDataElementsType } from "@/modules/data/types";
import DataFormRenderer from "../DataFormRenderer";

type Props = {
  dataType: ProfileDataElementsType;
};

export default function ProfileDataElement(props: Props) {
  const { dataType } = props;

  const rendererProp = getRendererProps(dataType);

  return <DataFormRenderer {...rendererProp} />;
}

const getRendererProps = (dataType: ProfileDataElementsType) => {
  let profileDataElementType = undefined;
  let isMultipleForms = false;
  let titleElementMultipleForms = undefined;

  switch (dataType) {
    case ProfileDataElements.personal:
      profileDataElementType = ProfileDataElements.personal;
      isMultipleForms = false;
      break;
    case ProfileDataElements.education:
      profileDataElementType = ProfileDataElements.education;
      isMultipleForms = true;
      titleElementMultipleForms = "institutionName";
      break;
    case ProfileDataElements.workExperience:
      profileDataElementType = ProfileDataElements.workExperience;
      isMultipleForms = true;
      titleElementMultipleForms = "companyName";
      break;
    case ProfileDataElements.hobbies:
      profileDataElementType= ProfileDataElements.hobbies;
      isMultipleForms = true;
      titleElementMultipleForms = "hobbyTitle";
      break;
    case ProfileDataElements.references:
      profileDataElementType= ProfileDataElements.references;
      isMultipleForms = true;
      titleElementMultipleForms = "companyName";
      break;
    case ProfileDataElements.certificates:
      profileDataElementType= ProfileDataElements.certificates;
      isMultipleForms = true;
      titleElementMultipleForms = "certificateName";
      break;
  }

  return {
    profileDataElementType,
    isMultipleForms,
    titleElementMultipleForms,
  };
};


