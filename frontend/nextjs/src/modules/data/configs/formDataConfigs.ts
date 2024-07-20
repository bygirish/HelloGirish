import {
  ConfigurableFormFieldDataType,
  ConfigurableFormFieldTypes,
} from "@/app/components/molecules/ConfigurableHookForm";
import * as yup from "yup";
import { ProfileDataElementsType } from "../types";
import { ProfileDataElements } from "../constants";

export const locationFormConfig = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const { address, city, country, postal } = initialData || {};

  return [
    {
      id: "address",
      label: "Address",
      initialValue: address || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "city",
      label: "City",
      initialValue: city || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "country",
      label: "Country",
      initialValue: country || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "postal",
      label: "Postal",
      initialValue: postal || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
  ];
};

export const personalInfoFormConfig = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const {
    firstName,
    lastName,
    gender,
    dob,
    phone,
    emailId,
    nationality,
    location,
    web,
    aboutMe,
  } = initialData || {};

  return [
    {
      id: "firstName",
      label: "First Name",
      initialValue: firstName || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "lastName",
      label: "Last Name",
      initialValue: lastName || "",
      validation: yup.string().required("Last Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
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
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 12, xl: 12 },
    },
    {
      id: "dob",
      label: "Date of Birth",
      initialValue: dob || "",
      validation: yup.date().required("Date of Birth is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "phone",
      label: "Phone number",
      initialValue: phone || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "emailId",
      label: "Email Address",
      initialValue: emailId || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },

    {
      id: "nationality",
      label: "Nationality",
      initialValue: nationality || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },

    {
      id: "web",
      label: "Web",
      initialValue: web || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },

    {
      id: "location",
      label: "Location",
      initialValue: location || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },

    {
      id: "aboutMe",
      label: "About Me",
      initialValue: aboutMe || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
  ];
};

export const educationFormConfig = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const {
    institutionName,
    studyField,
    degree,
    city,
    country,
    startTime,
    endTime,
    stillStudy,
    description,
  } = initialData || {};

  return [
    {
      id: "institutionName",
      label: "institution Name",
      initialValue: institutionName || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "studyField",
      label: "Field of Study",
      initialValue: studyField || "",
      validation: yup.string().required("Last Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "degree",
      label: "Degree",
      initialValue: degree || "",
      validation: yup
        .string()
        .oneOf(["male", "female"], "Please specify a valid gender")
        .required("Gender is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 12, xl: 12 },
    },
    {
      id: "city",
      label: "City",
      initialValue: city || "",
      validation: yup.date().required("Date of Birth is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "country",
      label: "Country",
      initialValue: country || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "startTime",
      label: "Start Time",
      initialValue: startTime || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "endTime",
      label: "End time",
      initialValue: endTime || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "stillStudy",
      label: "I currently study here",
      initialValue: stillStudy || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "description",
      label: "Description",
      initialValue: description || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
  ];
};

export const workExperienceFormConfig = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const {
    companyName,
    jobTitle,
    city,
    country,
    startTime,
    endTime,
    stillWorking,
    description,
  } = initialData || {};

  return [
    {
      id: "companyName",
      label: "Company Name",
      initialValue: companyName || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "jobTitle",
      label: "Job Title",
      initialValue: jobTitle || "",
      validation: yup.string().required("Last Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "city",
      label: "City",
      initialValue: city || "",
      validation: yup.date().required("Date of Birth is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "country",
      label: "Country",
      initialValue: country || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "startTime",
      label: "Start Time",
      initialValue: startTime || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "endTime",
      label: "End time",
      initialValue: endTime || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "stillWorking",
      label: "I currently work here",
      initialValue: stillWorking || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "description",
      label: "Description",
      initialValue: description || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
  ];
};

export const certificatesFormConfig = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const {
    certificateName,
    authority,
    certificationReference,
    dateOfCertification,
    description,
  } = initialData || {};

  return [
    {
      id: "certificateName",
      label: "Certificate Name",
      initialValue: certificateName || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "authority",
      label: "Authority",
      initialValue: authority || "",
      validation: yup.string().required("Last Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "certificationReference",
      label: "Certification URL / Code",
      initialValue: certificationReference || "",
      validation: yup.date().required("Date of Birth is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "dateOfCertification",
      label: "Date of Certification",
      initialValue: dateOfCertification || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "description",
      label: "Description",
      initialValue: description || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
  ];
};

export const companyReferencesFormConfig = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const { companyName, personName, contact1, contact2, referenceText } =
    initialData || {};

  return [
    {
      id: "companyName",
      label: "Company Name",
      initialValue: companyName || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 4 },
    },
    {
      id: "personName",
      label: "Person Name",
      initialValue: personName || "",
      validation: yup.string().required("Last Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "contact1",
      label: "Contact 1",
      initialValue: contact1 || "",
      validation: yup.date().required("Date of Birth is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "contact2",
      label: "Contact 2",
      initialValue: contact2 || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "referenceText",
      label: "Reference Text",
      initialValue: referenceText || "",
      validation: yup.string().required("Location is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
  ];
};

export const hobbyFormConfig = (
  initialData?: any
): ConfigurableFormFieldDataType[] => {
  const { hobbyIcon, hobbyTitle } = initialData || {};

  return [
    {
      id: "hobbyIcon",
      label: "Hobby Icon",
      initialValue: hobbyIcon || "",
      validation: yup.string().required("First Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
    {
      id: "hobbyTitle",
      label: "Hobby Name",
      initialValue: hobbyTitle || "",
      validation: yup.string().required("Last Name is required"),
      fieldType: ConfigurableFormFieldTypes.textInput,
      grid: { xs: 12, md: 12, lg: 6, xl: 6 },
    },
  ];
};


export const getFormConfigurations = (dataType: ProfileDataElementsType, initialData: any): ConfigurableFormFieldDataType[] => {

  // const existingData = profileData[dataType];

  initialData = initialData || {};

  switch (dataType) {
    case ProfileDataElements.personal:
      return personalInfoFormConfig(initialData);
    case ProfileDataElements.education:
      return educationFormConfig(initialData);
    case ProfileDataElements.workExperience:
      return workExperienceFormConfig(initialData);
    case ProfileDataElements.hobbies:
      return hobbyFormConfig(initialData);
    case ProfileDataElements.references:
      return companyReferencesFormConfig(initialData);
    case ProfileDataElements.certificates:
      return certificatesFormConfig(initialData);
    default:
      return [];
  }

}