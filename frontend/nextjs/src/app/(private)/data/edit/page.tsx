"use client";
import {
  Box,
  Tab,
  Tabs,
  Typography,
} from "@/app/components/atoms/index";
import { ProfileDataElements } from "@/modules/data/constants";
import React from "react";
import ProfileDataElement from "@/modules/data/components/ProfileDataElement";

type Props = {
  params: {
    id: string;
  };
};


const profileDataElements = [
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

export default function FillData(props: Props) {
  const onSubmit = (values: any, { setSubmitting }: any) => {
    setTimeout(() => {
      alert(JSON.stringify(values, null, 2));
      //   setSubmitting(false);
    }, 400);
  };

  const { id: profileId } = props.params;

  const [value, setValue] = React.useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };


  return (
    <Box
      sx={{
        flexGrow: 1,
        display: "flex",
        height: 'inherit'
      }}
    >
      <Tabs
        orientation="vertical"
        value={value}
        onChange={handleChange}
        aria-label="Vertical tabs example"
        sx={{ borderRight: 1, borderColor: "divider" }}
      >
        {profileDataElements.map((data, index) => {
          return (
            <Tab
              key={`tab-${data.key}-${index}`}
              label={data.text}
              id={`vertical-tab-${index}`}
              sx={{
                py: "20px",
                border: "1px ridge #ccc",
              }}
            />
          );
        })}
      </Tabs>
      {profileDataElements.map((data, index) => {
        return (
          <TabPanel
            key={`panel-${data.key}-${index}`}
            value={value}
            index={index}
          >
            <ProfileDataElement dataType={data.key}/>
          </TabPanel>
        );
      })}
    </Box>
  );
}



interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`vertical-tabpanel-${index}`}
      aria-labelledby={`vertical-tab-${index}`}
      {...other}
      style={{
        padding: 3, 
        width: '30%'
      }}
    >
      {value === index && (
        <Box>
          {children}
        </Box>
      )}
    </div>
  );
}
