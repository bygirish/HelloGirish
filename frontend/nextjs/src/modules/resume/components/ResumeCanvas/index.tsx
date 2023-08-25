import { Box } from "@/app/components/atoms";
import { ProfileDataElementsType } from "@/modules/data/types";

type Props = {};

export default function ResumeCanvas(props: Props) {
  const sections: ProfileDataElementsType[] = [
    "personal",
    "workExperience",
    "education",
    "certificates",
    "references",
  ];

  return (
    <Box sx={{
        width: '210mm', /* A4 width */
        minHeight: '297mm',
        margin: '20mm auto',
        padding: '10mm',
        backgroundColor: 'white',
        boxShadow: '0 0 5px rgba(0, 0, 0, 0.1)'
    }}>
      {sections.map((section, index) => {
        return (
          <Box
            key={index}
            sx={{
                height: '400px',
                width: '100mm',
                marginBottom: '15mm',
                pageBreakInside: 'avoid', /* Avoid breaking sections across pages */
                border: '1px solid',
            }}
          >
            {section}
          </Box>
        );
      })}
    </Box>
  );
}
