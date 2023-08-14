import { ProfileDataElements } from "@/modules/data/constants";
import { ProfileDataElementsType } from "@/modules/data/types";
import DataFormRenderer from "../DataFormRenderer";
import {
  certificatesFormConfig,
  companyReferencesFormConfig,
  educationFormConfig,
  hobbyFormConfig,
  personalInfoFormConfig,
  workExperienceFormConfig,
} from "@/modules/data/configs";
import { ConfigurableFormFieldDataType } from "@/app/components/molecules/ConfigurableHookForm";

type Props = {
  dataType: ProfileDataElementsType;
};

export default function ProfileDataElement(props: Props) {
  const { dataType } = props;

  const rendererProp = getRendererProps(dataType);

  return <DataFormRenderer {...rendererProp} />;
}

const getRendererProps = (dataType: ProfileDataElementsType) => {
  let formId = undefined;
  let configurationData: ConfigurableFormFieldDataType[] = [];
  let isMultipleForms = false;
  let titleElementMultipleForms = undefined;

  switch (dataType) {
    case ProfileDataElements.personal:
      formId = "personal-data-form";
      configurationData = personalInfoFormConfig({});
      isMultipleForms = false;
      break;
    case ProfileDataElements.education:
      formId = "education-data-form";
      configurationData = educationFormConfig({});
      isMultipleForms = true;
      titleElementMultipleForms = "institutionName";
      break;
    case ProfileDataElements.workExperience:
      formId = "work-experience-data-form";
      configurationData = workExperienceFormConfig({});
      isMultipleForms = true;
      titleElementMultipleForms = "companyName";
      break;
    case ProfileDataElements.hobbies:
      formId = "hobby-data-form";
      configurationData = hobbyFormConfig({});
      isMultipleForms = true;
      titleElementMultipleForms = "hobbyTitle";
      break;
    case ProfileDataElements.references:
      formId = "references-data-form";
      configurationData = companyReferencesFormConfig({});
      isMultipleForms = true;
      titleElementMultipleForms = "companyName";
      break;
    case ProfileDataElements.certificates:
      formId = "certificates-data-form";
      configurationData = certificatesFormConfig({});
      isMultipleForms = true;
      titleElementMultipleForms = "certificateName";
      break;
  }

  return {
    formId,
    configurationData,
    isMultipleForms,
    titleElementMultipleForms,
  };
};
